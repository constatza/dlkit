from collections.abc import Mapping, Sequence
from pathlib import Path

import mlflow
import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger
from pydantic import DirectoryPath, validate_call
from requests.compat import urlparse


class NumpyWriter(Callback):
    """Callback to accumulate multiple key-value pair predictions during prediction
    and write them to disk.

    The callback expects each predict_step output to be a dictionary with one or
    more key-value pairs, where each value is a torch.Tensor.
    """

    @validate_call
    def __init__(
        self, output_dir: DirectoryPath | None = None, filenames: Sequence[str] = ("predictions",)
    ) -> None:
        """Initialize the NumpyWriter callback.

        Args:
            output_dir (str): Directory path where the aggregated predictions will be saved.
        """
        super().__init__()
        self.output_dir = output_dir
        self._use_mlflow = False
        if self.output_dir is None:
            self.output_dir = Path(urlparse(mlflow.get_artifact_uri()).path.lstrip("/"))
            self._use_mlflow = True
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        # This dictionary will accumulate predictions across batches.
        # The keys are strings and values are lists of torch.Tensor.
        self._predictions: dict[str, list[torch.Tensor]] = {}
        self._filenames: Sequence[str] = filenames

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Mapping[str, torch.Tensor],
        batch: tuple,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Accumulate the outputs from each prediction batch.

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
            for i, (key, value) in enumerate(outputs.items()):
                write_key = self._filenames[i] if len(self._filenames) > i else key
                self._store_predictions(write_key, value)
        elif isinstance(outputs, list | tuple):
            for i, value in enumerate(outputs):
                self._store_predictions(self._filenames[i], value)
        elif isinstance(outputs, torch.Tensor):
            self._store_predictions(self._filenames[0], outputs)
        else:
            logger.error(f"Unexpected output type in NumpyWriter: {type(outputs)}")

    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """At the end of prediction, concatenate accumulated outputs for each key and write them to disk.

        Args:
            trainer (pl.Trainer): The current Trainer instance.
            pl_module (pl.LightningModule): The Lightning module.
        """
        if not self._predictions:
            logger.error("No predictions accumulated in NumpyWriter.")
            return

        # Process each key independently.
        for key, tensor_list in self._predictions.items():
            # Concatenate along the batch dimension.
            concatenated = torch.cat(tensor_list, dim=0).cpu().numpy()
            # Save all arrays into one compressed file (e.g. predictions.npz)
            output_path = Path(self.output_dir, f"{key}.npy")
            try:
                np.save(output_path, concatenated)
                if self._use_mlflow:
                    run_id = mlflow.active_run().info.run_id
                    mlflow.log_artifact(
                        str(output_path), artifact_path="predictions", run_id=run_id
                    )

                logger.info(f"Successfully saved output: {output_path}")
            except OSError as e:
                logger.error(f"Failed to save: {output_path}")
                logger.error(f"Error: {e}")
                continue

    def _store_predictions(self, key: str, value: torch.Tensor) -> None:
        """Store a prediction tensor for a given key.

        Args:
            key (str): The key under which the prediction is stored.
            value (torch.Tensor): The prediction tensor to store.
        """
        if not isinstance(value, torch.Tensor):
            logger.warning(f"Output for key '{key}' is not a torch.Tensor; skipping.")
            return
        if key not in self._predictions:
            self._predictions[key] = []
        self._predictions[key].append(value)
