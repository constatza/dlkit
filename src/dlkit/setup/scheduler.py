from torch import optim
from dlkit.settings import SchedulerSettings


def initialize_scheduler(
    scheduler: SchedulerSettings, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LRScheduler | None:

    scheduler_class = (
        getattr(optim.lr_scheduler, scheduler.name) if scheduler.name else None
    )

    return (
        scheduler_class(optimizer, **scheduler.to_dict_compatible_with(scheduler_class))
        if scheduler_class
        else None
    )
