import numpy as np

from sklearn.model_selection import train_test_split


def split_indices(
    indices: np.ndarray | list,
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 1,
):
    """
    Splits the dataset into train, validation, and test indices.
    """
    train_idx, test_plus_val_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state
    )
    val_idx, test_idx = train_test_split(
        test_plus_val_idx, test_size=val_size, random_state=random_state
    )
    return train_idx, val_idx, test_idx
