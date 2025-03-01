import matplotlib.pyplot as plt
import numpy as np

from dlkit.io.readers import read_study

study = read_study("./config.toml")
paths = study.paths

data = np.load(paths.input / "dataset.npz")
targets = data["targets"]
predictions = np.load(paths.names)


num_plots = 3

# random num_plots indices from axis 0
sample_idx = np.random.randint(0, targets.shape[0], num_plots)
# random num_plots indices from axis 1
dof_idx = np.random.randint(0, targets.shape[1], num_plots)


timesteps = np.arange(0, targets.shape[-1])
# common x-axis
fig, ax = plt.subplots(num_plots, 1, figsize=(10, 10), sharex=True)
for i, idx in enumerate(sample_idx):
    ax[i].plot(timesteps, targets[idx, dof_idx[i], :].T, label="Original")
    ax[i].plot(timesteps, predictions[idx, dof_idx[i], :].T, label="Predicted")
    ax[i].set_title(f"Sample {idx}")
    ax[i].legend()
    ax[i].set_xlabel("Timestep")
    ax[i].set_ylabel("P")

plt.show()
