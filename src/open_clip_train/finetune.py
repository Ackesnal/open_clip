import os
import re
import sys
import copy
import glob
import json
import math
import time
import random
import logging
import argparse
import subprocess
from tqdm import tqdm
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import wandb
except ImportError:
    wandb = None
    
try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None
    
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None
    
from open_clip import create_model_and_transforms, create_model, trace_model, get_tokenizer, create_loss
from open_clip import get_input_dtype, build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from open_clip_train.data import get_imagenet
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown, yang_lr
from open_clip_train.file_utils import pt_load, start_sync_process, remote_sync
from open_clip_train.precision import get_autocast

LATEST_CHECKPOINT_NAME = 'epoch_latest.pt'


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split('(\\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote: bool):
    if remote:
        result = subprocess.run(['aws', 's3', 'ls', path + '/'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def main(args):
    
    ##########################################################################################################
    # ↓↓ 1. Setup pre-finetuning configurations ↓↓ ###########################################################
    
    if not type(args) == argparse.Namespace:
        args = parse_args(args)
    
    # Setup cuda configs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Initialize distributed device environment
    if not args.optuna:
        device = init_distributed_device(args)
    else:
        device = args.device_info
    
    # Setup the name of the experiments
    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        if args.distributed:
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([date_str, 
                              f'model_{model_name_safe}', 
                              f'lr_{args.lr}', 
                              f'b_{args.batch_size}', 
                              f'j_{args.workers}', 
                              f'p_{args.precision}', 
                              'IMGNET_finetune'])
        
    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and (not resume_latest) and not args.optuna:
            print('Error. Experiment already exists. Use --name {} to specify a new experiment.')
            return -1
        
    # Setup text logger
    if not args.optuna:
        args.log_level = logging.DEBUG if args.debug else logging.INFO
        setup_logging(args.log_path, args.log_level)
    
    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, 'checkpoints')
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, 'tensorboard') if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    # Setup resuming from the latest checkpoint when available
    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, 'checkpoints')
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume_latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Check for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                     # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                 # Otherwise, don't try to resume
                resume_from = None
                
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No {LATEST_CHECKPOINT_NAME} found in {checkpoint_path}.')
                logging.info(f'Will try to load pretrained weight from {args.pretrained_weight}.')
                
        if args.distributed:
            # Sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from
        
    if args.copy_codebase:
        copy_codebase(args)
    
    # Start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # First make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # If all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')
    
    
    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    
    # Set deterministic seeds for reproducing experiment results    
    random_seed(args.seed, args.rank)
    
    # ↑↑ 1. Setup pre-finetuning configurations ↑↑ ###########################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 2. Initialize RePaCLIP (student) model ↓↓ ###########################################################
    
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)
        model_kwargs['init_logit_bias'] = -10
    
    # Build model
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model, 
        args.pretrained, 
        precision=args.precision, 
        device=device, 
        jit=args.torchscript, 
        force_quick_gelu=args.force_quick_gelu, 
        force_custom_text=args.force_custom_text, 
        force_patch_dropout=args.force_patch_dropout, 
        force_image_size=args.force_image_size, 
        image_mean=args.image_mean, 
        image_std=args.image_std, 
        image_interpolation=args.image_interpolation, 
        image_resize_mode=args.image_resize_mode, 
        aug_cfg=args.aug_cfg, 
        pretrained_image=args.pretrained_image, 
        output_dict=True, 
        cache_dir=args.cache_dir, 
        channel_idle=args.channel_idle, 
        idle_ratio=args.idle_ratio, 
        feature_norm=args.feature_norm, 
        slab=args.slab, 
        heuristic=args.idle_ratio_heuristic, 
        **model_kwargs
    )
    
    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)
    
    # Freeze text transformer during training when required
    if args.finetune_visual:
        for name, param in model.named_parameters():
            if 'visual' not in name and 'logit_scale' not in name:
                param.requires_grad = False
            elif 'embedding' in name or 'conv1' in name or 'ln_pre' in name:
                param.requires_grad = False
                
    if args.grad_checkpointing:
        model.set_grad_checkpointing()
    
    # Distribute model
    if args.distributed and (not args.horovod):
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True, **ddp_args)
    
    if is_master(args):
        logging.info('Model:')
        logging.info(f'{str(model)}')
        logging.info('Params:')
        params_file = os.path.join(args.logs, args.name, 'params.txt')
        with open(params_file, 'w') as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f'  {name}: {val}')
                f.write(f'{name}: {val}\n')
                
    # ↑↑ 2. Initialize RePaCLIP (student) model ↑↑ ###########################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 3. Initialize Vanilla CLIP (teacher) model ↓↓ #######################################################
    
    # Build teacher model
    teacher_model = create_model(
        args.model, 
        args.pretrained, 
        precision=args.precision, 
        device=device, 
        jit=args.torchscript, 
        force_quick_gelu=args.force_quick_gelu, 
        force_custom_text=args.force_custom_text, 
        force_patch_dropout=args.force_patch_dropout, 
        force_image_size=args.force_image_size, 
        pretrained_image=args.pretrained_image, 
        output_dict=True, 
        cache_dir=args.cache_dir, 
        **model_kwargs
    )
    
    # Freeze the whole teacher model as it does not need to be trained
    # No DDP for a model with no module needing grad
    teacher_model.eval()
    for name, param in teacher_model.named_parameters():
        param.requires_grad = False
        
    # ↑↑ 3. Initialize Vanilla CLIP (teacher) model ↑↑ #######################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 4. Load dataset ↓↓ ##################################################################################
    
    data_loader_train = get_imagenet(args, (preprocess_train, preprocess_val), 'train')
    data_loader_val = get_imagenet(args, (preprocess_train, preprocess_val), 'val')
    data = {'train': data_loader_train, 
            'imagenet-val': data_loader_val}
    
    # ↑↑ 4. Load dataset ↑↑ ##################################################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 5. Initialize optimizer, scaler, lr scheduler and loss ↓↓ ###########################################
    
    optimizer = None
    scaler = None
    scheduler = None
    
    if args.imagenet_train:
        assert not args.trace, 'Cannot train with traced model'
        
        # Initialize optimizer
        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), 'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(model, timm_opt, lr=args.lr, weight_decay=args.wd, eps=args.eps, **opt_kwargs)
        else:
            exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or ('bias' in n) or ('logit_scale' in n)
            include = lambda n, p: not exclude(n, p)
            named_parameters = list(model.named_parameters())
            
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            if opt == 'adamw':
                optimizer = optim.AdamW([{'params': gain_or_bias_params, 'weight_decay': 0.0, 'name': 'no_decay'}, 
                                         {'params': rest_params, 'weight_decay': args.wd, 'name': 'decay'}],
                                        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps)
            else:
                assert False, f'Unknown optimizer {opt}'
                
        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}')
            
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        
        # Initialize scaler
        if args.precision == 'amp':
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()
                
        # Initialize lr scheduler
        total_steps = math.ceil(len(data['train'].dataloader) / args.accum_freq) * args.epochs
        if args.lr_scheduler == 'yang':
            freeze_epoch = model.module.visual.transformer.layers
            freeze_steps = math.ceil(len(data['train'].dataloader) / args.accum_freq) * freeze_epoch
            scheduler = yang_lr(optimizer, args.lr, freeze_steps, total_steps)
        elif args.lr_scheduler == 'cosine':
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == 'const':
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == 'const-cooldown':
            assert args.epochs_cooldown is not None, 'Please specify the number of cooldown epochs for this lr schedule.'
            cooldown_steps = data['train'].dataloader.num_batches // args.accum_freq * args.epochs_cooldown
            scheduler = const_lr_cooldown(optimizer, args.lr, args.warmup, total_steps, cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)
            
        # Initialize loss function
        # We only use self-distillation with the vanilla model
        # Avoid using args.distill (conflict with normal KD)
        args.self_distill_loss = True
        args.distill = False
        loss = create_loss(args)
    else:
        if is_master(args):
            logging.info('Training dataset not found.')
        return -1
    
    # ↑↑ 5. Initialize optimizer, scaler, lr scheduler and loss ↑↑ ###########################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 6. Load pretrained/resumed checkpoint for student model ↓↓ ##########################################
    
    assert args.pretrained_weight is not None, 'Pretrained weight should be given using --pretrained_weight when finetuning.'
    
    # Load weight for student model
    if args.resume is None:
        # Without available args.resume, load from pretrained weights
        checkpoint = pt_load(args.pretrained_weight, map_location='cpu')
    else:
        # Otherwise, load from resumed checkpoint
        checkpoint = pt_load(args.resume, map_location='cpu')
    
    # Load state_dict (with modification to vanilla CLIP)
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    if args.distributed and (not next(iter(sd.items()))[0].startswith('module')):
        sd = {f'module.{k}': v for k, v in sd.items()}
    sd = {k.replace('ln_2', 'mlp.ln'): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False if args.resume is None else True)
    
    if args.resume and 'epoch' in checkpoint:
        start_epoch = checkpoint['epoch']
        if is_master(args):
            logging.info(f"=> Student resuming checkpoint '{args.resume}' (epoch {start_epoch - 1}).")
    else:
        start_epoch = 0
        if is_master(args):
            logging.info(f"=> Student resuming checkpoint '{args.pretrained_weight}'.")
    
    # If start_epoch > 0 and model performs channel_idle, then adjust its idling layers when args.yang
    if start_epoch > 0 and args.channel_idle:
        # Yang Method: Progressively add channel idle mechanism from the deepest layer to the shallowest layer
        if args.yang:
            if args.distributed:
                total_layer = model.module.visual.transformer.layers
            else:
                total_layer = model.visual.transformer.layers
            progressive_epochs = total_layer - 1 # Total number of progressive layers
            layers = [i for i in range(max(0, progressive_epochs - (start_epoch - 1)), total_layer)]
            if args.distributed:
                model.module.visual.adapt_idle(layers)
            else:
                model.visual.adapt_idle(layers)
            if is_master(args):
                logging.info(f'Yang method: Adjust finetuned layers to {layers} w.r.t. resumed weights')
        
        # Yang Method with Layer Freeze: Freeze the layers without channel idle
        if args.yang and args.yang_freeze:
            for name, param in model.named_parameters():
                if 'visual' in name:
                    if 'ln_post' in name:
                        param.requires_grad = True
                    elif 'mask' in name:
                        param.requires_grad = False
                    else:
                        names = name.split('.')
                        if len(names) >= 6:
                            layer = int(names[4])
                            if layer > min(layers):
                                param.requires_grad = True
                            elif layer == min(layers) and 'mlp' in name and ('mask' not in name):
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                        else:
                            param.requires_grad = False
                else:
                    param.requires_grad = False
                    
            # Generate new optimizer according to the model
            exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or ('bias' in n) or ('logit_scale' in n)
            include = lambda n, p: not exclude(n, p)
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['name'] == 'no_decay':
                    optimizer.param_groups[i]['params'] = gain_or_bias_params
                else:
                    optimizer.param_groups[i]['params'] = rest_params
    
    # Load optimizer and scaler when available
    if 'optimizer' in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scaler' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])
        
    # ↑↑ 6. Load pretrained/resumed checkpoint for student model ↑↑ ##########################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 7. Load pretrained weight for teacher model ↓↓ ######################################################
    
    checkpoint = pt_load(args.pretrained_weight, map_location='cpu')
    sd = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items()}
    sd = {k.replace('ln_2', 'mlp.ln'): v for k, v in sd.items()}
    teacher_model.load_state_dict(sd, strict=True)
    if is_master(args):
        logging.info(f"=> Teacher resuming checkpoint '{args.pretrained_weight}'.")
    
    # ↑↑ 7. Load pretrained weight for teacher model ↑↑ ######################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 8. Generate channel idle masks w.r.t. heuristics ↓↓ #################################################
    
    if args.channel_idle and args.generate_mask:
        if args.resume:
            logging.info('Resume from checkpoint. No need for generating new masks.')
        else:
            if is_master(args):
                logging.info('Pre-calculating channel distribution...')
            
            with torch.no_grad():
                finetune_data = get_imagenet(args, (preprocess_train, preprocess_val), 'val').dataloader
                for batch in tqdm(finetune_data) if is_master(args) else finetune_data:
                    images, _ = batch
                    images = images.to(device=torch.device(args.device), non_blocking=True)
                    if args.distributed:
                        model.module.visual(images, record_positive=True)
                    else:
                        model.visual(images, record_positive=True)
                if args.distributed:
                    model.module.visual.generate_mask()
                else:
                    model.visual.generate_mask()
                    
            torch.distributed.barrier()
            if is_master(args):
                logging.info('Generated masks for each FFN layer!')
            
    # ↑↑ 8. Generate channel idle masks w.r.t. heuristics ↑↑ #################################################
    ##########################################################################################################        
            
      
    ##########################################################################################################
    # ↓↓ 9. Setup pretraining configs, wandb, etc. ↓↓ ########################################################
    
    # Determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, 'Please install tensorboard.'
        writer = tensorboard.SummaryWriter(args.tensorboard_path)
    
    # (TBD) Start wandb session
    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data['train'].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data['val'].dataloader.num_samples
        wandb.init(project=args.wandb_project_name, 
                   name=args.name, 
                   id=args.name, 
                   notes=args.wandb_notes, 
                   tags=[], 
                   resume='auto' if args.resume == 'latest' else None, 
                   config=vars(args))
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')
        
    # ↑↑ 9. Setup pretraining configs, wandb, etc. ↑↑ ########################################################
    ##########################################################################################################    
    
    
    ##########################################################################################################
    # ↓↓ 10. Setup zero-shot classifier and evaluate ↓↓ ######################################################
    
    # Generate tokenizer
    if is_master(args):
        logging.info('Building tokenizer')
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    torch.distributed.barrier()
    
    # Generate classifier based on text transformer
    if is_master(args):
        logging.info('Building zero-shot classifier')
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        classifier = build_zero_shot_classifier(
            model.module if args.distributed else model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )
    torch.distributed.barrier()
    
    # Evaluate student model's performance before finetuning with distillation
    # if is_master(args):
    #     logging.info(f"=> Evaluate student's performance.")
    # evaluate(model, classifier, data, 0, args, tb_writer=writer, tokenizer=tokenizer)
    # torch.distributed.barrier()
    
    # Evaluate teacher model's performance before finetuning with distillation
    # if is_master(args):
    #     logging.info(f"=> Evaluate teacher's performance.")
    # evaluate(teacher_model, classifier, data, 0, args, tb_writer=writer, tokenizer=tokenizer)
    # torch.distributed.barrier()
    
    # ↑↑ 10. Setup zero-shot classifier and evaluate ↑↑ ######################################################
    ##########################################################################################################  
    
    
    ##########################################################################################################
    # ↓↓ 11. Finetune on ImageNet ↓↓ #########################################################################
    
    if is_master(args):
        logging.info(f'')
        logging.info('============================= Finetuning start =============================')
        logging.info(f'')
    
    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')
            
        # Set model (student) to train and teacher model to eval
        model.train()
        teacher_model.eval()
        
        # Yang Method: Progressively add channel idle mechanism from the deepest layer to the shallowest layer
        if args.yang:
            if args.distributed:
                total_layer = model.module.visual.transformer.layers
            else:
                total_layer = model.visual.transformer.layers
            progressive_epochs = total_layer - 1
            layers = [i for i in range(max(0, progressive_epochs - epoch), total_layer)]
            if is_master(args):
                logging.info(f'Yang method: Finetuning {layers} layers')
            if args.distributed:
                model.module.visual.adapt_idle(layers)
            else:
                model.visual.adapt_idle(layers)
                
        # Yang Method with Layer Freeze: Freeze the layers without channel idle
        if args.yang and args.yang_freeze:
            for name, param in model.named_parameters():
                if 'visual' in name:
                    if 'ln_post' in name:
                        param.requires_grad = True
                    elif 'mask' in name:
                        param.requires_grad = False
                    else:
                        names = name.split('.')
                        if len(names) >= 6:
                            layer = int(names[4])
                            if layer > min(layers):
                                param.requires_grad = True
                            elif layer == min(layers) and 'mlp' in name and ('mask' not in name):
                                param.requires_grad = True
                            else:
                                param.requires_grad = False
                        else:
                            param.requires_grad = False
                elif 'logit_scale' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            
            # Re-initialize optimizer with more parameters added to the optimizer
            exclude = lambda n, p: p.ndim < 2 or 'bn' in n or 'ln' in n or ('bias' in n) or ('logit_scale' in n)
            include = lambda n, p: not exclude(n, p)
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            for i in range(len(optimizer.param_groups)):
                if optimizer.param_groups[i]['name'] == 'no_decay':
                    optimizer.param_groups[i]['params'] = gain_or_bias_params
                else:
                    optimizer.param_groups[i]['params'] = rest_params
                    
        # Finetune
        finetune_one_epoch(model, teacher_model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=writer)
        completed_epoch = epoch + 1
        
        # Evaluate 
        if any((v in data for v in ['val', 'imagenet-val', 'imagenet-v2'])):
            metrics = evaluate(model, classifier, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
            
        # Save log and checkpoint
        if args.save_logs:
            checkpoint_dict = {'epoch': completed_epoch, 
                               'name': args.name, 
                               'state_dict': model.state_dict(), 
                               'optimizer': optimizer.state_dict()
                               }
            
            if scaler is not None:
                checkpoint_dict['scaler'] = scaler.state_dict()
                
            if (completed_epoch == args.epochs or (args.save_frequency > 0 and completed_epoch % args.save_frequency == 0)) and not args.optuna:
                torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f'epoch_{completed_epoch}.pt'))
                
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f'epoch_{completed_epoch - 1}.pt')
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)
            
            if args.save_most_recent:
                tmp_save_path = os.path.join(args.checkpoint_path, 'tmp.pt')
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME) 
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)
    # End for
                
    # ↑↑ 11. Finetune on ImageNet ↑↑ #########################################################################
    ##########################################################################################################
    
    
    ##########################################################################################################
    # ↓↓ 12. End finetuning ↓↓ ###############################################################################
    
    if args.wandb and is_master(args):
        wandb.finish()
        
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    
    if is_master(args) and args.optuna:
        return metrics['imagenet-zeroshot-val-top1']
    
    # ↑↑ 12. End finetuning ↑↑ ###############################################################################
    ##########################################################################################################
    
    

def finetune_one_epoch(model, teacher_model, data, loss, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    data['train'].set_epoch(epoch)
    dataloader = data['train'].dataloader
    num_batches_per_epoch = math.ceil(len(dataloader) / args.accum_freq)
    
    # Setup SLAB per-step adjust
    if args.slab:
        total_step = max((args.epochs - 15), 5) * num_batches_per_epoch
    elif args.feature_norm in ['LayerNorm', 'LN', 'ln', 'layernorm']:
        gamma = 1
    else:
        gamma = 0
    
    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    for i, samples in enumerate(dataloader):
        i_accum = i // args.accum_freq
        step = num_batches_per_epoch * epoch + i_accum
        
        if not args.skip_scheduler:
            scheduler(step)
        
        # Update SLAB normalization gamma
        if args.slab and i % args.accum_freq == 0:
            slab_step = num_batches_per_epoch * (epoch - 15) + i_accum
            gamma = max(1 - slab_step / total_step, 0)
            model.module.adapt_gamma(gamma)
        
        images, labels = samples
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        data_time_m.update(time.time() - end)
        
        with autocast():
            model_out = model(images, None)
            teacher_model_out = teacher_model(images, None)
            losses = loss(model_out['image_features'], 
                          model_out['logit_scale'], 
                          teacher_model_out['image_features'], 
                          teacher_model_out['logit_scale'], 
                          output_dict=True)
            
        total_loss = losses['distill_loss'] / args.accum_freq
        backward(total_loss, scaler)
        
        if (i + 1) % args.accum_freq > 0 and i + 1 < len(dataloader):
            continue
        
        if scaler is not None:
            if args.horovod:
                optimizer.synchronize()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                with optimizer.skip_synchronize():
                    scaler.step(optimizer)
            else:
                if args.grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
                scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            optimizer.step()
            
        optimizer.zero_grad()
        
        # Note: we clamp to 4.6052 = ln(100), as in the original paper.
        with torch.no_grad():
            unwrap_model(model).logit_scale.clamp_(0, math.log(100))
            
        batch_time_m.update(time.time() - end)
        end = time.time()
        
        batch_count = i_accum + 1
        if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
            # Calculate completed percentage
            batch_size = images.shape[0]
            num_samples = (i + 1) * batch_size * args.world_size
            samples_per_epoch = len(dataloader) * batch_size * args.world_size
            percent_complete = 100.0 * num_samples / samples_per_epoch
            
            # Calculate speed
            samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
            
            # Calculate rest time
            rest_seconds = math.ceil((samples_per_epoch - num_samples) / samples_per_second)
            
            # Log losses
            for key, val in losses.items():
                if key not in losses_m:
                    losses_m[key] = AverageMeter()
                losses_m[key].update(val.item(), batch_size)
            
            loss_log = " ".join(
                [
                    f"{loss_name.capitalize()}: {loss_m.val:.5g} ({loss_m.avg:.5g})" 
                    for loss_name, loss_m in losses_m.items()
                ]
            )
            
            # Log latest training info
            logging.info(
                f"Train Epoch: {epoch} [{num_samples}/{samples_per_epoch} ({percent_complete:.0f}%)] | "
                f"Data (t): {data_time_m.avg:.3f} | "
                f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:.3f}/s, {samples_per_second_per_gpu:.3f}/s/gpu | "
                f"Est: {(rest_seconds // 3600):02d}:{(rest_seconds % 3600 // 60):02d}:{(rest_seconds % 60):02d} | "
                f"LR: {optimizer.param_groups[0]['lr']:.8f} | "
                f"SLAB Gamma: {gamma:.4f} | " + loss_log)
            
            log_data = {'data_time': data_time_m.val, 
                        'batch_time': batch_time_m.val, 
                        'samples_per_second': samples_per_second, 
                        'samples_per_second_per_gpu': samples_per_second_per_gpu, 
                        'lr': optimizer.param_groups[0]['lr']}
            
            log_data.update({name: val.val for name, val in losses_m.items()})
            log_data = {'train/' + name: val for name, val in log_data.items()}
            
            if tb_writer is not None:
                for name, val in log_data.items():
                    tb_writer.add_scalar(name, val, step)
            if args.wandb:
                assert wandb is not None, 'Please install wandb.'
                log_data['step'] = step
                wandb.log(log_data, step=step)
            batch_time_m.reset()
            data_time_m.reset()
            
            
            
def evaluate(model, classifier, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return metrics
    model.eval()
    
    zero_shot_metrics = zero_shot_eval(model, classifier, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics



def zero_shot_eval(model, classifier, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        if type(model) == torch.nn.parallel.DistributedDataParallel:
            model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)
        
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results



def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]



def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, 'code')
    if os.path.exists(new_code_path):
        print(f'Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment.')
        return -1
    print(f'Copying codebase to {new_code_path}')
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print('Done copying code.')
    return 1


if __name__ == '__main__':
    main(sys.argv[1:])