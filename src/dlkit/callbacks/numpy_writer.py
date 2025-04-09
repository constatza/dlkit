import torch
import numpy as np

from collections.abc import Mapping
from lightning.pytorch import Callback, Trainer, LightningModule
from loguru import logger

from pydantic import validate_call, DirectoryPath


class NumpyWriter(Callback):
    """
    Callback to accumulate multiple key-value pair predictions during prediction
    and write them to disk as a single .npz file.

    The callback expects each predict_step output to be a dictionary with one or
    more key-value pairs, where each value is a torch.Tensor.

    Attributes:
        output_path (str): The directory where the predictions will be saved.
    """

    @validate_call
    def __init__(self, output_dir: DirectoryPath) -> None:
        """
        Initialize the NumpyWriter callback.

        Args:
            output_dir (str): Directory path where the aggregated predictions will be saved.
        """
        super().__init__()
        self.output_dir = output_dir
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        # This dictionary will accumulate predictions across batches.
        # The keys are strings and values are lists of torch.Tensor.
        self._predictions: dict[str, list[torch.Tensor]] = {}

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """
        Accumulate the outputs from each prediction batch.

        Args:
            trainer (pl.Trainer): The current Trainer instance.
            pl_module (pl.LightningModule): The Lightning module.
            outputs (Any): The output from the predict_step. Expected to be a dict.
            batch (Any): The current batch.
            batch_idx (int): Index of the batch.
            dataloader_idx (int, optional): Index of the dataloader (default is 0).
        """
        # Ensure the outputs are a mapping.
        if isinstance(outputs, Mapping):
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor):
                    logger.warning(
                        f"Output for key '{key}' is not a torch.Tensor; skipping."
                    )
                    continue
                if key not in self._predictions:
                    self._predictions[key] = []
                self._predictions[key].append(value)
        else:
            logger.warning(
                "Expected outputs from predict_step to be a dict; skipping accumulation."
            )

    def on_predict_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        """
        At the end of prediction, concatenate accumulated outputs for each key and write them to disk.

        Args:
            trainer (pl.Trainer): The current Trainer instance.
            pl_module (pl.LightningModule): The Lightning module.
        """
        if not self._predictions:
            logger.warning("No predictions accumulated in NumpyWriter.")
            return

        # Process each key independently.
        for key, tensor_list in self._predictions.items():
            # Concatenate along the batch dimension.
            concatenated = torch.cat(tensor_list, dim=0).cpu().numpy()
            # Save all arrays into one compressed file (e.g. predictions.npz)
            output_path = self.output_dir / f"{key}.npy"
            try:
                np.save(output_path, concatenated)
                logger.info(f"Successfully saved: {output_path}")
            except IOError as e:
                logger.error(f"Failed to save: {output_path}")
                logger.error(f"Error: {e}")
                continue
