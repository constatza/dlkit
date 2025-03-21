from torch import optim
from dlkit.settings.classes import SchedulerSettings, OptimizerSettings
from dlkit.utils.system_utils import filter_kwargs


def initialize_scheduler(
    scheduler: SchedulerSettings, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LRScheduler | None:

    scheduler_class = (
        getattr(optim.lr_scheduler, scheduler.name) if scheduler.name else None
    )

    return (
        scheduler_class(optimizer, **filter_kwargs(scheduler.model_dump()))
        if scheduler_class
        else None
    )
