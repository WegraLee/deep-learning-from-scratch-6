def get_lr(it, max_lr, warmup_iters, max_iters):
    # 워밍업： 0 -> max_lr
    if it < warmup_iters:
        return max_lr * (it / warmup_iters)

    # 어닐링： max_lr -> 0
    if it < max_iters:
        progress = (it - warmup_iters) / (max_iters - warmup_iters)
        return max_lr * (1.0 - progress)

    return 0.0