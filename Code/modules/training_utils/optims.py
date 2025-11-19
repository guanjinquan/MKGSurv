from torch.optim import Adam, AdamW
import torch

# Optimizer参数:
# optimizer: 'Adam' / 'AdamW'
# backbone_lr: 0.0001
# learning_rate: 0.001
# weight_decay: 0.0001

# Scheduler参数:
# scheduler: 'CosineAnnealingLR' / 'CosineAnnealingLR_warmup' / 'OneCycleLR'
# num_epochs: 100


def GetOptimizer(args, model):
    import math
    import torch

    # 兼容 DDP 包装：优先使用 model.module
    raw_model = getattr(model, "module", model)

    # 尝试读取 backbone params（若模型实现了 get_backbone_params）
    try:
        training_params = list(raw_model.get_params())
    except Exception as e:
        print(f"[WARN] get_backbone_params() not implemented for model {type(model)}. Using all params for optimizer.")
        print("Exception:", e, flush=True)
        training_params = []

    # 所有参数（原始模型）
    if not training_params:
        training_params = list(raw_model.parameters())

    # 只统计 requires_grad=True 的参数（即会被 optimizer 更新的）
    def params_size_mb(param_list, only_trainable=True):
        total_bytes = 0
        for p in param_list:
            if p is None:
                continue
            if only_trainable and (not getattr(p, "requires_grad", True)):
                continue
            total_bytes += p.numel() * p.element_size()
        return total_bytes / (1024 ** 2)

    total_trainable_mb = params_size_mb(training_params, only_trainable=True)

    # 打印 summary
    print(f"[PARAMS] Trainable params (MB): total={total_trainable_mb:.3f} MB")

    # Helper: 过滤掉 requires_grad=False 的参数，避免把不可训练参数加入 optimizer
    def filter_trainable(param_list):
        return [p for p in param_list if getattr(p, "requires_grad", True)]

    # --- 构造 optimizer ---
    # For other branches, include backbone and others separately (if present)
    training_params = filter_trainable(training_params)

    # Fallback: if no backbone detected, just optimize all trainable params as a single group
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam([
            {'params': training_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW([
            {'params': training_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD([
            {'params': training_params, 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ], momentum=0.9)
    else:
        raise ValueError("optimizer not supported")

    # 再次打印 optimizer 内实际 param 大小（有助于确认 optimizer 中包含哪些参数）
    print("Num of param groups:", len(optimizer.param_groups)) # self.optimizer.param_groups
    opt_param_count = 0
    opt_bytes = 0
    for g in optimizer.param_groups:
        for p in g['params']:
            opt_param_count += p.numel()
            opt_bytes += p.numel() * p.element_size()
    opt_mb = opt_bytes / (1024 ** 2)
    print(f"[OPTIMIZER] param groups: {len(optimizer.param_groups)}, total_params_in_optimizer={opt_param_count}, approx_size={opt_mb:.3f} MB")

    return optimizer

def GetScheduler(args, optim):
    if args.scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs, eta_min=1e-8)
    elif args.scheduler == 'CosineAnnealingLR_warmup':
        assert args.num_epochs % 2 == 0, "num_epochs must be even"
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs // 2, eta_min=1e-8)
    elif args.scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
                optim, 
                max_lr=[args.backbone_lr, args.learning_rate], 
                epochs=args.num_epochs, 
                steps_per_epoch=1, 
                anneal_strategy='cos'
            )
    else:
        raise ValueError("scheduler not supported") 