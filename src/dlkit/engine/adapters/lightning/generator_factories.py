"""Generator factory implementations for stochastic training reproducibility.

Generator factories produce (or decline to produce) a ``torch.Generator``
for each batch. They are injected into the wrapper so training and validation
reproducibility strategies can be swapped without modifying the wrapper.
"""

import torch


class NullGeneratorFactory:
    """Return ``None`` for every batch — uses PyTorch's global RNG.

    Implements ``IGeneratorFactory``.  The default for training: maximum
    throughput, no per-batch seed management.
    """

    def __call__(self, batch_idx: int) -> torch.Generator | None:
        """Return None (global RNG).

        Args:
            batch_idx: Batch index (ignored).

        Returns:
            ``None`` — callers use the global RNG.
        """
        return None


class DeterministicGeneratorFactory:
    """Seed a new generator per batch for fully reproducible stochastic ops.

    Implements ``IGeneratorFactory``.  Recommended for validation and testing
    so that stochastic supervision (e.g. time sampling) is identical across runs.

    Args:
        base_seed: Base seed; each batch gets seed ``base_seed + batch_idx``.
        device: Device for the generator (default ``"cpu"``).
    """

    def __init__(self, base_seed: int = 42, device: str | torch.device = "cpu") -> None:
        self._base_seed = base_seed
        self._device = torch.device(device)

    def __call__(self, batch_idx: int) -> torch.Generator:
        """Return a freshly seeded generator for this batch.

        Args:
            batch_idx: Batch index — added to base_seed for uniqueness.

        Returns:
            Seeded ``torch.Generator`` on the configured device.
        """
        gen = torch.Generator(device=self._device)
        gen.manual_seed(self._base_seed + batch_idx)
        return gen


class FixedGeneratorFactory:
    """Always return the same pre-built generator — useful for replay / debugging.

    Implements ``IGeneratorFactory``.

    Args:
        generator: Pre-built generator to return every call.
    """

    def __init__(self, generator: torch.Generator) -> None:
        self._generator = generator

    def __call__(self, batch_idx: int) -> torch.Generator:
        """Return the fixed generator.

        Args:
            batch_idx: Batch index (ignored).

        Returns:
            The pre-built generator.
        """
        return self._generator
