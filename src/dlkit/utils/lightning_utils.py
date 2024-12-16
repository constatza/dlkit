from lightning.pytorch.tuner import Tuner


def tune_lr(trainer, model, *args, **kwargs):

    tuner = Tuner(
        trainer,
    )

    lr_finder = tuner.lr_find(
        model,
        *args,
        **kwargs,
    )

    fig = lr_finder.plot(suggest=True, show=True)

    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()

    # update hparams of the model
    model.hparams.lr = new_lr
    return model
