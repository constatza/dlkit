import matplotlib.pyplot as plt
import numpy as np

from dlkit.io.readers import load_config

config: dict = load_config("./p.toml")
paths = config["paths"]


features = np.load(paths["features"])

predictions = np.load(paths["predictions"])
# stack first two axes
predictions = predictions.reshape(-1, predictions.shape[-2], predictions.shape[-1])
timesteps = np.arange(0, features.shape[-1])


num_plots = 3

# random num_plots indices from axis 0
sample_idx = np.random.randint(0, features.shape[0], num_plots)
# random num_plots indices from axis 1
dof_idx = np.random.randint(0, features.shape[1], num_plots)


timesteps = np.arange(0, features.shape[-1])
# common x-axis
fig, ax = plt.subplots(num_plots, 1, figsize=(10, 10), sharex=True)
for i, idx in enumerate(sample_idx):
    ax[i].plot(timesteps, features[idx, dof_idx[i], :].T, label="Original")
    ax[i].plot(timesteps, predictions[idx, dof_idx[i], :].T, label="Predicted")
    ax[i].set_title(f"Sample {idx}")
    ax[i].legend()
    ax[i].set_xlabel("Timestep")
    ax[i].set_ylabel("P")
# common x-axis

from dlkit.postprocessing import plot_residuals

# plot_residuals(predictions.ravel(), features.ravel())

plt.show()

# visualize latent space
# add figure with three 3d subplots
# fig = plt.figure(figsize=(15, 10))
# for i in range(3):
#     ax = fig.add_subplot(1, 3, i + 1, projection="3d")
#     scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=parameters[:, i])
#     ax.set_xlabel("Latent 1")
#     ax.set_ylabel("Latent 2")
#     ax.set_zlabel("Latent 3")
#     # add colorbar
#     cbar = plt.colorbar(scatter, ax=ax, location="bottom")
#     cbar.set_label(f"Bio-Parameter {i + 1}")
# fig.suptitle("Latent Space")
# plt.show()
