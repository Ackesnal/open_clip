import os
import sys
import torch
import optuna
import logging
from datetime import datetime
from open_clip_train.params import parse_args
from open_clip_train.logger import setup_logging
from open_clip_train.finetune import main as finetune
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object

args = None
  
def objective(trial):
    torch.distributed.barrier()
    
    if is_master(args):
        lr = trial.suggest_float('lr', 1e-6, 1e-4)
        wd = trial.suggest_float('wd', 0.0, 0.5)
        epochs = trial.suggest_int('epochs', 15, 50)
        yang_freeze = trial.suggest_categorical('yang_freeze', [True, False])
        beta2 = trial.suggest_float('beta2', 0.9, 0.99)
        args.lr = lr
        args.wd = wd
        args.epochs = epochs
        args.yang_freeze = yang_freeze
        args.beta2 = beta2
        
    args.lr = broadcast_object(args, args.lr)
    args.wd = broadcast_object(args, args.wd)
    args.epochs = broadcast_object(args, args.epochs)
    args.wd = broadcast_object(args, args.wd)
    args.yang_freeze = broadcast_object(args, args.yang_freeze)
    args.beta2 = broadcast_object(args, args.beta2)
    
    torch.distributed.barrier()
        
    acc = finetune(args)
    
    torch.distributed.barrier()
    return acc
                    
if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    
    # Setup cuda configs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    # Initialize distributed device environment
    device = init_distributed_device(args)
    args.device_info = device
    
    if args.name is None:
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
        if args.distributed:
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([f'optuna_',
                              date_str, 
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
        if os.path.exists(args.log_path) and not resume_latest:
            print('Error. Experiment already exists. Use --name {} to specify a new experiment.')
    
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    
    args.optuna = True
    args.save_most_recent = False
    args.resume = None
    args.save_frequency = -1
    
    if is_master(args):
        study = optuna.create_study(study_name=args.name,  direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=args.optuna_ntrials)
        if is_master(args):
            print(f"Best hyperparameters: {study.best_params}")
            print(f"Best validation accuracy: {study.best_value}")
    else:
        cnt = 0
        while cnt < args.optuna_ntrials:
            objective(None)
            cnt += 1
    