"""Tests for generator factory implementations.

Covers:
- NullGeneratorFactory: always returns None
- DeterministicGeneratorFactory: returns a seeded torch.Generator per batch
- DeterministicGeneratorFactory: different batch_idx → different seeds
- DeterministicGeneratorFactory: same batch_idx + same base_seed → reproducible output
- FixedGeneratorFactory: always returns the same generator object
"""

from __future__ import annotations

import pytest
import torch

from dlkit.core.models.wrappers.generator_factories import (
    DeterministicGeneratorFactory,
    FixedGeneratorFactory,
    NullGeneratorFactory,
)

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
_BASE_SEED: int = 42
_BATCH_IDX_A: int = 0
_BATCH_IDX_B: int = 1


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def null_factory() -> NullGeneratorFactory:
    """NullGeneratorFactory instance.

    Returns:
        NullGeneratorFactory.
    """
    return NullGeneratorFactory()


@pytest.fixture
def deterministic_factory() -> DeterministicGeneratorFactory:
    """DeterministicGeneratorFactory with a fixed base seed.

    Returns:
        DeterministicGeneratorFactory seeded at _BASE_SEED.
    """
    return DeterministicGeneratorFactory(base_seed=_BASE_SEED)


@pytest.fixture
def pre_built_generator() -> torch.Generator:
    """Pre-built seeded generator for FixedGeneratorFactory.

    Returns:
        Seeded torch.Generator.
    """
    gen = torch.Generator()
    gen.manual_seed(_BASE_SEED)
    return gen


@pytest.fixture
def fixed_factory(pre_built_generator: torch.Generator) -> FixedGeneratorFactory:
    """FixedGeneratorFactory wrapping a pre-built generator.

    Args:
        pre_built_generator: Pre-built generator fixture.

    Returns:
        FixedGeneratorFactory.
    """
    return FixedGeneratorFactory(pre_built_generator)


# ===========================================================================
# NullGeneratorFactory
# ===========================================================================


def test_null_generator_factory_returns_none(null_factory: NullGeneratorFactory) -> None:
    """NullGeneratorFactory always returns None regardless of batch_idx.

    Args:
        null_factory: NullGeneratorFactory fixture.
    """
    assert null_factory(0) is None
    assert null_factory(99) is None


def test_null_factory_multiple_calls(null_factory: NullGeneratorFactory) -> None:
    """NullGeneratorFactory returns None for arbitrary batch indices.

    Args:
        null_factory: NullGeneratorFactory fixture.
    """
    for idx in range(10):
        assert null_factory(idx) is None


# ===========================================================================
# DeterministicGeneratorFactory
# ===========================================================================


def test_deterministic_factory_returns_generator(
    deterministic_factory: DeterministicGeneratorFactory,
) -> None:
    """DeterministicGeneratorFactory returns a torch.Generator instance.

    Args:
        deterministic_factory: DeterministicGeneratorFactory fixture.
    """
    result = deterministic_factory(_BATCH_IDX_A)
    assert isinstance(result, torch.Generator)


def test_deterministic_factory_different_batch_idx(
    deterministic_factory: DeterministicGeneratorFactory,
) -> None:
    """Different batch_idx values produce generators with different states.

    Produces one random sample from each generator; they should differ.

    Args:
        deterministic_factory: DeterministicGeneratorFactory fixture.
    """
    gen_a = deterministic_factory(_BATCH_IDX_A)
    gen_b = deterministic_factory(_BATCH_IDX_B)
    sample_a = torch.rand(1, generator=gen_a)
    sample_b = torch.rand(1, generator=gen_b)
    assert not torch.equal(sample_a, sample_b)


def test_deterministic_factory_same_seed_reproducible(
    deterministic_factory: DeterministicGeneratorFactory,
) -> None:
    """Same batch_idx + same base_seed → identical random samples.

    Args:
        deterministic_factory: DeterministicGeneratorFactory fixture.
    """
    gen1 = deterministic_factory(_BATCH_IDX_A)
    gen2 = deterministic_factory(_BATCH_IDX_A)
    sample1 = torch.randn(4, generator=gen1)
    sample2 = torch.randn(4, generator=gen2)
    assert torch.equal(sample1, sample2)


def test_deterministic_factory_custom_device() -> None:
    """DeterministicGeneratorFactory respects the device argument.

    Creates a factory with device='cpu' and verifies the generator's
    device attribute is correctly set.
    """
    factory = DeterministicGeneratorFactory(base_seed=0, device="cpu")
    gen = factory(0)
    assert gen.device == torch.device("cpu")


# ===========================================================================
# FixedGeneratorFactory
# ===========================================================================


def test_fixed_factory_returns_same_generator(
    fixed_factory: FixedGeneratorFactory, pre_built_generator: torch.Generator
) -> None:
    """FixedGeneratorFactory returns the identical generator object every call.

    Args:
        fixed_factory: FixedGeneratorFactory fixture.
        pre_built_generator: Generator that was injected into the factory.
    """
    result_a = fixed_factory(_BATCH_IDX_A)
    result_b = fixed_factory(_BATCH_IDX_B)
    assert result_a is pre_built_generator
    assert result_b is pre_built_generator


def test_fixed_factory_ignores_batch_idx(
    fixed_factory: FixedGeneratorFactory,
) -> None:
    """FixedGeneratorFactory returns the same object for arbitrary batch indices.

    Args:
        fixed_factory: FixedGeneratorFactory fixture.
    """
    first = fixed_factory(0)
    for idx in range(1, 10):
        assert fixed_factory(idx) is first
