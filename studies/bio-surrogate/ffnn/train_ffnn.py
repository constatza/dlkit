import shutil

import torch
import torch.nn as nn
from lightning.pytorch.callbacks import TQDMProgressBar, ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.trainer import Trainer

from dlkit.data.file_datamodule import FileDataModule
from dlkit.io.readers import read_study
from dlkit.networks.ffnns import FeedForwardNN
from dlkit.transforms.chaining import Pipeline
from dlkit.transforms import NumpyToTensor, MinMaxScaler
from dlkit.utils.system_utils import tune_lr

config_file = "./config.toml"
config = read_study(config_file)

input_size = 3
output_size = 4
tune = False
continue_training = False
rebuild_dataset = False
num_epochs = 4000
batch_size = 512
lr = 5e-2
layers = [input_size, 400, 500, output_size]

torch.set_float32_matmul_precision("high")

paths = config.paths
root_dir = paths.workdir

logger = CSVLogger(paths.logs)

callbacks = [
    ModelSummary(max_depth=2),
    TQDMProgressBar(),
    ModelCheckpoint(
        monitor="val_loss",
        dirpath=paths.checkpoints,
        filename="best-checkpoint",
        save_top_k=2,
        mode="min",
        save_last=True,
        every_n_epochs=10,
    ),
]

feature_transform = Pipeline(
    MinMaxScaler(),
    NumpyToTensor(),
)
target_transforms = Pipeline(
    NumpyToTensor(),
)

data_module = FileDataModule(
    save_dir=paths.input,
    paths=(paths.features, paths.targets),
    transforms=(feature_transform, target_transforms),
    rebuild_dataset=rebuild_dataset,
    batch_size=batch_size,
    test_size=0.3,
    indices_path=paths.metadata,
)

data_module.setup(stage="fit")

trainer = Trainer(
    default_root_dir=root_dir,
    max_epochs=num_epochs,
    accelerator="gpu",
    callbacks=callbacks,
    enable_progress_bar=True,
    reload_dataloaders_every_n_epochs=10,
    logger=logger,
)

model = FeedForwardNN(layers=layers, activation=nn.GELU(), lr=lr)

# Pick point based on plot, or get suggestion

model = tune_lr(trainer, model, datamodule=data_module) if tune else model

if continue_training:
    trainer.fit(model, datamodule=data_module, ckpt_path="last")
else:
    # delete the checkpoint directory
    shutil.rmtree(paths.checkpoints, ignore_errors=True)
    shutil.rmtree(paths.logs, ignore_errors=True)
    trainer.fit(model, datamodule=data_module)

data_module.setup(stage="test")
test_score = trainer.test(model, datamodule=data_module)

data_module.setup(stage="predict")
output = trainer.predict(model, datamodule=data_module)

predictions = torch.cat(output, dim=0).cpu().numpy()
# predictions = data_module.target_transforms.inverse_transform(predictions)
# targets = data_module.target_transforms.inverse_transform(data_module.targets)
targets = data_module.targets.cpu().numpy()

import matplotlib.pyplot as plt


x_hat = predictions[:, 0]
y_hat = predictions[:, 1]
x = targets[:, 0]
y = targets[:, 1]
# hollow circles for predictions
plt.scatter(
    x_hat,
    y_hat,
    label="Predictions",
)
plt.scatter(x, y, label="Targets", marker="+", c="r")
plt.legend()
plt.colorbar()
plt.xlabel("latent 1")
plt.ylabel("latent 2")
plt.title("FFNN fit")
plt.show()
