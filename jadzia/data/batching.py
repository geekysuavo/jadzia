"""Batching functionality."""

# Standard
from collections.abc import Iterable, Sequence
from typing import cast

# Third party
import jax
import numpy as np

type Value = bool | int | float | np.ndarray
"""Values accepted for batching."""


class Batcher:
    """Build batched data from sequences of elements."""

    def __init__(self, concat: Iterable[str] | None = None):
        """
        Initialize the batcher.

        Args:
            concat: Output keys that should be batched using [numpy.concat][]
                instead of the default [numpy.stack][].
        """
        self._concat = dict.fromkeys(concat or (), 0)

    def __call__(self, elements: Sequence[dict[str, Value]]) -> dict[str, np.ndarray]:
        """
        Batch a sequence of elements.

        Args:
            elements: Sequence of dictionaries to batch into a single dictionary.

        Returns:
            Dictionary of batched data.
        """
        return jax.tree.map_with_path(self._batch_leaf, *elements)

    def _batch_leaf(
        self,
        path: tuple[jax.tree_util.DictKey],
        *values: Value,
    ) -> np.ndarray:
        """
        Batch a sequence of pytree leaf values.

        Args:
            path: Tree path indicating which output key is being batched.
            values: Input values to batch.

        Returns:
            Array of batched values.
        """
        key = path[0].key
        if key not in self._concat:
            return np.stack(values)

        assert all(isinstance(value, np.ndarray) for value in values)
        values = cast(tuple[np.ndarray, ...], values)

        concat_size = sum(value.shape[0] for value in values)
        pad_to_size = max(self._concat[key], concat_size)
        self._concat[key] = pad_to_size

        if concat_size < pad_to_size:
            shape = (pad_to_size - concat_size, *values[0].shape[1:])
            values = (*values, np.zeros(shape))

        return np.concat(values)


__all__ = ["Batcher"]
