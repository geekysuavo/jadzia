"""Batching functionality."""

# Standard
from collections.abc import Iterable, Sequence

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
        self._concat = set(concat or ())
        self._pad: dict[str, int] = {}

    def __call__(self, elements: Sequence[dict[str, Value]]) -> dict[str, np.ndarray]:
        """
        Batch a sequence of elements.

        Each element may contain [bool][], [int][], [float][] or [numpy.ndarray][]
        values. When values are arrays, they may differ in shape along the first
        dimension only.

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
        arrays = tuple(np.asarray(value) for value in values)
        if key in self._concat:
            return self._concat_leaf(key, arrays)
        else:
            return self._stack_leaf(key, arrays)

    def _concat_leaf(self, key: str, values: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Batch a sequence of values by concatenation followed by optional padding.

        Args:
            key: Output key.
            values: Input values to concatenate.

        Returns:
            Array of concatenated values.
        """
        concat_size = sum(value.shape[0] for value in values)
        pad_to_size = max(self._pad.get(key, 0), concat_size)
        self._pad[key] = pad_to_size

        if concat_size < pad_to_size:
            shape = (pad_to_size - concat_size, *values[0].shape[1:])
            values = (*values, np.zeros(shape))

        return np.concat(values)

    def _stack_leaf(self, key: str, values: tuple[np.ndarray, ...]) -> np.ndarray:
        """
        Batch a sequence of values by optional padding followed by stacking.

        Args:
            key: Output key.
            values: Input values to stack.

        Returns:
            Array of stacked values.
        """
        shapes = {value.shape for value in values}
        if len(shapes) == 1:
            if shapes.pop() == ():
                return np.stack(values).reshape(-1, 1)
            else:
                return np.stack(values)

        sizes = tuple(value.shape[0] for value in values)
        pad_to_size = max(self._pad.get(key, 0), max(sizes))
        self._pad[key] = pad_to_size

        def pad_fn(value: np.ndarray) -> np.ndarray:
            return np.pad(value, ((0, pad_to_size - value.shape[0]),))

        return np.stack(tuple(map(pad_fn, values)))


__all__ = ["Batcher"]
