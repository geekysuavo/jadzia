"""Tests of batching functionality."""

# Standard
import random

# Third party
import numpy as np
import pytest

# Dax
from jadzia.data.batching import Batcher


@pytest.mark.parametrize("batch_size", [1, 8, 32])
def test_first_call(batch_size):
    """Fixed-shape arrays are stacked correctly into batched arrays."""
    rng = np.random.default_rng()
    elements = [
        {
            "a": random.randint(0, 8),
            "b": random.normalvariate(),
            "w": rng.integers(255, size=(index + 1, 4), dtype=np.uint8),
            "x": rng.integers(8, size=(), dtype=np.int16),
            "y": rng.normal(size=(2,)).astype(np.float16),
            "z": rng.normal(size=(3, 5)),
        }
        for index in range(batch_size)
    ]

    batch_fn = Batcher(concat=("w",))
    batch = batch_fn(elements)
    assert isinstance(batch, dict)
    assert set(batch) == {"a", "b", "w", "x", "y", "z"}

    assert batch["a"].shape == (batch_size,)
    assert batch["b"].shape == (batch_size,)
    assert batch["w"].shape == (batch_size * (batch_size + 1) // 2, 4)
    assert batch["x"].shape == (batch_size,)
    assert batch["y"].shape == (batch_size, 2)
    assert batch["z"].shape == (batch_size, 3, 5)

    assert batch["a"].dtype == np.int64
    assert batch["b"].dtype == np.float64
    assert batch["w"].dtype == np.uint8
    assert batch["x"].dtype == np.int16
    assert batch["y"].dtype == np.float16
    assert batch["z"].dtype == np.float64


def test_concat():
    """Batchers pad concatenated values to the maximum size seen thus far."""
    batch_fn = Batcher(concat=("s",))

    first = [{"s": np.full((2,), 1.1)}, {"s": np.full((3,), 2.2)}]
    second = [{"s": np.full((1,), 3.3)}, {"s": np.full((2,), 4.4)}]
    third = [{"s": np.full((3,), 5.5)}, {"s": np.full((5,), 6.6)}]
    fourth = [{"s": np.full((2,), 7.7)}, {"s": np.full((4,), 8.8)}]

    masks = (
        np.array([1, 1, 1, 1, 1], dtype=np.bool),
        np.array([1, 1, 1, 0, 0], dtype=np.bool),
        np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.bool),
        np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=np.bool),
    )

    batches = list(map(batch_fn, (first, second, third, fourth)))
    assert all(isinstance(batch, dict) for batch in batches)
    assert all(set(batch) == {"s"} for batch in batches)

    arrays = tuple(batch["s"] for batch in batches)
    sizes = tuple(array.shape[0] for array in arrays)

    assert sizes == (5, 5, 8, 8)
    for array, mask in zip(arrays, masks):
        np.testing.assert_equal(array != 0, mask)


def test_stack():
    """Batchers pad stacked values to the maximum size seen thus far."""
    batch_fn = Batcher()

    first = [{"s": np.full((2,), 1.1)}, {"s": np.full((3,), 2.2)}]
    second = [{"s": np.full((1,), 3.3)}, {"s": np.full((2,), 4.4)}]
    third = [{"s": np.full((3,), 5.5)}, {"s": np.full((5,), 6.6)}]
    fourth = [{"s": np.full((2,), 7.7)}, {"s": np.full((4,), 8.8)}]

    masks = (
        np.array([[1, 1, 0], [1, 1, 1]], dtype=np.bool),
        np.array([[1, 0, 0], [1, 1, 0]], dtype=np.bool),
        np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=np.bool),
        np.array([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=np.bool),
    )

    batches = list(map(batch_fn, (first, second, third, fourth)))
    assert all(isinstance(batch, dict) for batch in batches)
    assert all(set(batch) == {"s"} for batch in batches)

    arrays = tuple(batch["s"] for batch in batches)
    sizes = tuple(array.shape[1] for array in arrays)
    batch_sizes = tuple(array.shape[0] for array in arrays)

    assert batch_sizes == (2, 2, 2, 2)
    assert sizes == (3, 3, 5, 5)
    for array, mask in zip(arrays, masks):
        np.testing.assert_equal(array != 0, mask)
