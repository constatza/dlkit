import torch
import torch.nn as nn
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint, ModelSummary
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from dlkit.data.file_datamodule import FileDataModule
from dlkit.io.readers import read_study
from dlkit.networks.ffnns import FeedForwardNN
from dlkit.transforms import NumpyToTensor, MinMaxScaler
from dlkit.transforms.chaining import Pipeline

torch.set_float32_matmul_precision("medium")
lr = 1e-2
EPOCHS = 1500
layers = [12, 100, 1000, 100, 1]
dropout = 0
batch_norm = False
layer_norm = True

config_file = "./config.toml"
config = read_study(config_file)
paths = config.paths

checkpointer = ModelCheckpoint(
    monitor="val_loss",
    dirpath=paths.checkpoints,
    filename="best-checkpoint",
    save_top_k=1,
    mode="min",
    save_last=True,
    every_n_epochs=10,
)

logger = TensorBoardLogger(save_dir=paths.output, name="logs")

callbacks = [
    ModelSummary(max_depth=2),
    RichProgressBar(leave=True),
    checkpointer,
]

feature_transform = Pipeline(
    NumpyToTensor(),
    MinMaxScaler(),
)

preprocessors = (lambda x: x, lambda x: x[-5, :].reshape(-1, 1))
target_transforms = Pipeline(
    NumpyToTensor(),
    MinMaxScaler(),
)
data_module = FileDataModule(
    save_dir=paths.output,
    paths=(paths.features, paths.targets),
    transforms=(feature_transform, target_transforms),
    test_size=0.3,
    val_size=0.5,
    batch_size=500,
    preprocessors=preprocessors,
)


trainer = Trainer(
    default_root_dir=config.paths.output,
    max_epochs=EPOCHS,
    accelerator="gpu",
    callbacks=callbacks,
    enable_progress_bar=True,
    reload_dataloaders_every_n_epochs=10,
    deterministic=True,
    logger=logger,
)

model = FeedForwardNN(
    layers=layers,
    activation=nn.Tanh(),
    lr=lr,
    dropout=dropout,
    batch_norm=batch_norm,
    layer_norm=layer_norm,
)

#
trainer.fit(model, datamodule=data_module)
#
model = FeedForwardNN.load_from_checkpoint(checkpointer.best_model_path)
print(checkpointer.best_model_path)
trainer.test(model, datamodule=data_module)

predictions = trainer.predict(model, datamodule=data_module)
