from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger
from pydantic import DirectoryPath, validate_call

from dlkit.tools.io.url_utils import get_url_path


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
            self.output_dir, self._use_mlflow = self._resolve_default_output_dir()
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
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
            if self.output_dir is None:
                logger.error("Output directory is None, cannot save predictions")
                continue
            output_path = Path(self.output_dir) / f"{key}.npy"
            try:
                np.save(output_path, concatenated)
                if self._use_mlflow:
                    import mlflow

                    current_run = mlflow.active_run()
                    if current_run is not None:
                        mlflow.log_artifact(str(output_path), artifact_path="predictions")

                logger.debug(f"Successfully saved output: {output_path}")
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

    @staticmethod
    def _resolve_default_output_dir() -> tuple[Path, bool]:
        """Resolve the default output directory for predictions.

        Checks whether an MLflow run is currently active. If so, returns the
        run's local artifact directory so that prediction arrays are written
        alongside other run artifacts. Falls back to ``<cwd>/predictions``
        when no run is active or when the artifact URI is not a local
        ``file://`` path.

        Returns:
            tuple[Path, bool]: A ``(path, use_mlflow)`` pair where ``use_mlflow``
                is ``True`` only when the path was derived from an active MLflow
                run's artifact store.

        Warning:
            In MLflow 3.x, calling ``mlflow.get_artifact_uri()`` **without an
            active run auto-creates a new run** and initializes the tracking
            store, potentially writing ``mlflow.db`` to the current working
            directory. This method guards against that side-effect by checking
            ``mlflow.active_run()`` before calling ``get_artifact_uri()``.
        """
        try:
            import mlflow

            if mlflow.active_run() is None:
                return Path.cwd() / "predictions", False
            artifact_uri = mlflow.get_artifact_uri()
            if artifact_uri and artifact_uri.startswith("file://"):
                return Path(get_url_path(artifact_uri).lstrip("/")), True
        except Exception:
            pass

        return Path.cwd() / "predictions", False
