def log_parameter_count(model, logger=None):
    """Log trainable/total parameter counts broken down by top-level module.

    Args:
        model: nn.Module (the full detector).
        logger: Optional logger. Falls back to print if None.
    """
    log = logger.info if logger else print

    def _count(module):
        total = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        return total, trainable

    total_all, train_all = _count(model)

    # Named top-level children → group into semantic blocks
    groups = {
        'img_backbone': 'Image Backbone',
        'img_neck': 'Image Neck (FPN)',
        'bev_constructor': 'BEV Constructor',
        'bbox_head': 'TE Detection Head',
        'pts_bbox_head': 'Lane Head (TopoLogicHead)',
    }

    sep = '-' * 70
    lines = [
        '',
        sep,
        f'{"Component":<35} {"Total":>12} {"Trainable":>12} {"Frozen":>12}',
        sep,
    ]

    accounted = 0
    for attr, label in groups.items():
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        t, tr = _count(mod)
        accounted += t
        lines.append(f'{label:<35} {t:>12,} {tr:>12,} {t - tr:>12,}')

    # Catch anything not in the named groups
    remainder = total_all - accounted
    if remainder > 0:
        lines.append(f'{"Other":<35} {remainder:>12,}')

    lines.append(sep)
    lines.append(f'{"TOTAL":<35} {total_all:>12,} {train_all:>12,} {total_all - train_all:>12,}')
    lines.append(f'  Trainable ratio: {train_all / total_all * 100:.1f}%')
    lines.append(sep)

    log('\n'.join(lines))
