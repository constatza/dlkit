import numpy as np
import torch
from sklearn.model_selection import train_test_split


def reshaper(func):
    """Decorator that scales data with format (num_samples, num_channels, num_dofs, num_timesteps)."""

    def wrapper(dataset, *args, **kwargs):

        # Reshape data to (num_samples, num_channels*num_dofs* num_timesteps)

        original_shape = dataset.shape
        num_features = np.prod(original_shape[1:])
        reshaped_dataset = dataset.reshape(original_shape[0], num_features)
        # Scale data
        reshaped_dataset = func(reshaped_dataset, *args, **kwargs)
        # Reshape data back to (num_samples, num_channels, num_dofs, num_timesteps)
        if isinstance(reshaped_dataset, tuple) or isinstance(reshaped_dataset, list):
            output = [d.reshape(-1, *original_shape[1:]) for d in reshaped_dataset]
        else:
            output = reshaped_dataset.reshape(original_shape)
        return output

    return wrapper


def scale_timeseries_torch(dataset: torch.Tensor, means=None, stds=None):
    """Scale the data using a standard scaler."""
    # Scale data
    # assuming dataset is of shape (num_samples, num_channels, num_features, num_timesteps)
    if means is None:
        means = torch.mean(dataset, dim=(0, -1), keepdim=True)
    if stds is None:
        stds = torch.std(dataset, dim=(0, -1), keepdim=True)

    return (dataset - means) / stds, means, stds


def scale_timeseries_numpy(dataset: np.ndarray, means=None, stds=None):
    """Scale the data using a standard scaler."""
    # Scale data
    # assuming dataset is of shape (num_samples, num_channels, num_features, num_timesteps)
    if means is None:
        means = np.mean(dataset, axis=(0, -1), keepdims=True)
    if stds is None:
        stds = np.std(dataset, axis=(0, -1), keepdims=True)

    return (dataset - means) / stds, means, stds


def scale_timeseries(dataset, *args, **kwargs):
    """Scale the data using a standard scaler."""
    if isinstance(dataset, torch.Tensor):
        return scale_timeseries_torch(dataset, *args, **kwargs)
    else:
        return scale_timeseries_numpy(dataset, *args, **kwargs)


def unscale_timeseries(dataset, means, stds):
    """Unscale the data using a standard scaler."""
    # Unscale data
    return dataset * stds + means


@reshaper
def split_data(dataset, *args, **kwargs):
    """Split the data into training and testing sets."""
    # Split data
    return train_test_split(dataset, *args, **kwargs)


def format_data(solutions_path, dofs_to_keep_path):
    """Import the data from the path and keep only the dofs specified."""
    # Load the dofs to keep
    import os

    dirname = os.path.dirname(solutions_path)
    solutions = np.load(solutions_path).T
    # dofs to keep
    dofs = np.loadtxt(dofs_to_keep_path).astype(int)

    solutions = np.transpose(solutions, (2, 0, 1))
    solutions = solutions[:, :, :-8]
    solutions = solutions[:, dofs, :]
    solutions = solutions[:, np.newaxis, :, :]
    np.save(os.path.join(dirname, "formatted_solutions.npy"), solutions)
    print(solutions.shape)
    return solutions


def unscaled_predict(model, targets, means, stds):
    predictions = model(targets).detach()
    return unscale_timeseries(predictions, means, stds)


def unscaled_predict_with_targets(model, targets, means, stds):
    predictions = unscaled_predict(model, targets, means, stds)
    targets = unscale_timeseries(targets, means, stds).float().detach()
    return predictions, targets
