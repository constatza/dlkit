import lightning
import numpy as np
import torch
from lightning import Trainer

from dlkit.data.file_datamodule import FileDataModule
from dlkit.io.readers import read_study
from dlkit.metrics import nrmse
from dlkit.networks.caes import BasicCAE
from dlkit.networks.ffnns import FeedForwardNN
from dlkit.transforms.chaining import Pipeline
from dlkit.transforms import MinMaxScaler, NumpyToTensor, StandardScaler

batch_size = 64
rebuild_dataset = True


class ChainedModel(lightning.LightningModule):
    def __init__(self, *models):
        super(ChainedModel, self).__init__()
        self.models = models

    def forward(self, x):
        latent = self.models[0].forward(x)
        x = self.models[1].decode(latent)
        return x

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        y = batch[1]
        y_hat = self.predict_step(batch, batch_idx, dataloader_idx)
        loss = self.loss(y_hat, y)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch[0]
        y_hat = self.forward(x)
        return y_hat

    def loss(self, y_hat, y):
        return nrmse(y_hat, y)


def predict(paths):
    feature_transform = Pipeline(
        NumpyToTensor(),
        MinMaxScaler(),
    )
    target_transforms = Pipeline(
        NumpyToTensor(),
        StandardScaler(dim=(0, -1)),
    )

    data_module = FileDataModule(
        save_dir=paths.input,
        paths=(paths.targets, paths.targets),
        transforms=(feature_transform, target_transforms),
        rebuild_dataset=rebuild_dataset,
        batch_size=batch_size,
        test_size=0.3,
        indices_path=paths.metadata,
    )

    mynetwork = FeedForwardNN.load_from_checkpoint(
        paths.checkpoints_ffnn / "best-checkpoint.ckpt"
    )

    second_network = BasicCAE.load_from_checkpoint(
        paths.checkpoints_cae / "best-checkpoint.ckpt"
    )

    model = ChainedModel(mynetwork, second_network)

    model.eval()
    model.freeze()
    trainer = Trainer(
        default_root_dir=paths.output,
        accelerator="gpu",
    )

    data_module.setup(stage="fit")
    data_module.setup(stage="test")
    trainer.test(model, datamodule=data_module)
    predictions = trainer.predict(model, datamodule=data_module)
    trainer.save_checkpoint(paths.checkpoints / "model.ckpt")

    stacked_predictions = torch.stack(predictions).squeeze()
    predictions = data_module.target_transforms.inverse(stacked_predictions)
    return predictions.cpu().numpy(), data_module.targets.cpu().numpy()


def main():
    from matplotlib import pyplot as plt

    config_file = "./config.toml"
    config = read_study(config_file)
    predictions, targets = predict(config.paths)
    np.save(config.paths.output / "predictions.npy", predictions)

    sample = np.random.randint(0, predictions.shape[0])
    dof = np.random.randint(0, predictions.shape[1])
    x = predictions[sample, dof, :]
    y = targets[sample, dof, :]

    fig, ax = plt.subplots()
    ax.plot(x, label="Predicted")
    ax.plot(y, label="True")
    ax.legend()
    ax.set_xlabel("Timestep")
    ax.set_ylabel("P")
    ax.grid(True)
    ax.set_title(f"Total Workflow: Test Sample {sample}, DOF {dof}")
    plt.show()
    fig.savefig(config.paths.figures / f"p_total_sample{sample}_dof{dof}.png", dpi=400)


if __name__ == "__main__":
    main()
