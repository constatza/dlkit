from torch import optim


def initialize_scheduler(config, optimizer):

    scheduler_name = config.pop("name", None)
    scheduler_class = (
        getattr(optim.lr_scheduler, scheduler_name) if scheduler_name else None
    )

    return scheduler_class(optimizer, **config) if scheduler_class else None
