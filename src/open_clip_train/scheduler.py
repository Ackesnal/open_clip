import math


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def const_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            lr = base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def const_lr_cooldown(optimizer, base_lr, warmup_length, steps, cooldown_steps, cooldown_power=1.0, cooldown_end_lr=0.):
    def _lr_adjuster(step):
        start_cooldown_step = steps - cooldown_steps
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            if step < start_cooldown_step:
                lr = base_lr
            else:
                e = step - start_cooldown_step
                es = steps - start_cooldown_step
                # linear decay if power == 1; polynomial decay otherwise;
                decay = (1 - (e / es)) ** cooldown_power
                lr = decay * (base_lr - cooldown_end_lr) + cooldown_end_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def finetune_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
            for param_group in optimizer.param_groups:
                if "repa" in param_group["name"]:
                    
                    param_group["lr"] = lr
                else:
                    param_group["lr"] = 0.0
            # print(f"LR for RePa parts is {lr}, LR for other parts is {0.0}")
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
            for param_group in optimizer.param_groups:
                if "repa" in param_group["name"]:
                    param_group["lr"] = lr
                else:
                    param_group["lr"] = 0.01 * lr
            # print(f"LR for RePa parts is {lr}, LR for other parts is {0.0}")
        return 
    
    # def _lr_adjuster(step):
    #     if step < warmup_length:
    #         lr = _warmup_lr(base_lr, warmup_length, step)
    #         for param_group in optimizer.param_groups:
    #             layer = int(param_group["name"].split("_")[-1])
    #             exp = int(param_group["name"].split("_")[-2])
    #             if layer == -1:
    #                 param_group["lr"] = lr
    #             else:
    #                 param_group["lr"] = lr * (0.7 ** exp)
    #             # print(f"Layer {layer}: {param_group['lr']}", "\n\n")
    #         # print(f"LR for RePa parts is {lr}, LR for other parts is {0.0}")
    #     else:
    #         e = step - warmup_length
    #         es = steps - warmup_length
    #         lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
    #         for param_group in optimizer.param_groups:
    #             layer = int(param_group["name"].split("_")[-1])
    #             exp = int(param_group["name"].split("_")[-2])
    #             if layer == -1:
    #                 param_group["lr"] = lr
    #             else:
    #                 param_group["lr"] = lr * (0.7 ** exp)
    #             # print(f"Layer {layer}: {param_group['lr']}", "\n\n")
    #         # print(f"LR for RePa parts is {lr}, LR for other parts is {0.0}")
    #     return 

    return _lr_adjuster


def yang_lr(optimizer, base_lr, freeze_steps, steps):
    def _lr_adjuster(step):
        if step <= freeze_steps:
            lr = base_lr
        else:
            e = step - freeze_steps
            es = steps - freeze_steps
            lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster