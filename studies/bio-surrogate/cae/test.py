from pathlib import Path

import numpy as np
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from dlkit.io.readers import read_toml
from dlkit.networks.caes import BasicCAE


num_layers = 4
kernel_size = 7
stride = 3
hidden_size = 3

config_path = "config.toml"
config = read_toml(config_path)
root_dir = config["paths"]["output"]
root_dir = Path(root_dir)
# predict
u = torch.from_numpy(np.load(dataset_path)).float()


trainer = Trainer()
model.eval()
with torch.no_grad():
    latent_data = model.encode(u)
    trainer.test(model, dataloaders=test_loader)

# save
print(latent_data.shape)
np.save(root_dir / "latent_data.npy", latent_data.detach().numpy())
