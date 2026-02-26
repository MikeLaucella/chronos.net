"""
lr_decay.py

The parameter group decay patterns for segformer.
"""

def param_groups_seg(model, base_lr: float=1e-4):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    decay = []
    no_decay = []
    head = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if 'head' in n or 'classifier' in n:
            head.append(p)
        elif 'norm' in n or 'embed' in n or 'patch_embeddings' in n:
            no_decay.append(p)
        else:
            decay.append(p)

    return [
        {"params": decay, "lr": base_lr, "weight_decay": 0.01},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        {"params": head, "lr": base_lr * 10, "lr_scale": 10, "weight_decay": 0.01}
    ]
