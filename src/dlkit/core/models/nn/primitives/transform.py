from typing import Any
import torch
from lightning.pytorch import Callback
from dlkit.core.training.transforms.chain import TransformChain
from dlkit.tools.config import ModelComponentSettings
from dlkit.tools.utils.torch_utils import dataloader_to_xy


class TransformMixin(Callback):
    def __init__(self, settings: ModelComponentSettings) -> None:
        super().__init__()  # important for cooperative MRO
        feature_transforms = getattr(settings, "feature_transforms")
        feature_shape = None
        if getattr(settings, "shape", None) is not None:
            feature_shape = getattr(settings.shape, "x", None)
        if feature_shape is None:
            msg = "TransformMixin requires settings.shape.x to be defined"
            raise ValueError(msg)
        self.features_chain = TransformChain(feature_transforms, input_shape=feature_shape)
        target_shape = None
        if getattr(settings, "shape", None) is not None:
            target_shape = getattr(settings.shape, "y", None)
        is_autoencoder = getattr(settings, "is_autoencoder", False)
        if not is_autoencoder and target_shape is None:
            msg = "TransformMixin requires settings.shape.y for non-autoencoder models"
            raise ValueError(msg)

        target_transforms = getattr(settings, "target_transforms")
        self.targets_chain = (
            TransformChain(target_transforms, input_shape=target_shape or feature_shape)
            if not is_autoencoder
            else self.features_chain
        )

    def forward(
        self, x: torch.Tensor, *args: Any, apply_target_inverse_chain: bool = False, **kwargs: Any
    ) -> torch.Tensor:
        x = self.features_chain(x)
        out = super().forward(x, *args, **kwargs)  # routes into LightningModule or your model
        return self.targets_chain.inverse_transform(out) if apply_target_inverse_chain else out

    def on_train_start(self) -> None:
        # move chains to device, fit transforms
        self.features_chain = self.features_chain.to(self.device)
        self.targets_chain = self.targets_chain.to(self.device)
        dl = self.trainer.datamodule.train_dataloader()
        x, y = dataloader_to_xy(dl)
        x = x.to(self.device)
        self.features_chain.fit(x)
        if not self.settings.is_autoencoder:
            y = y.to(self.device)
            self.targets_chain.fit(y)
