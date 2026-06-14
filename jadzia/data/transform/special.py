"""Special transformation classes."""

# Standard
from collections.abc import Iterable
from typing import Any

# Third party
import grain
import numpy as np

# Dax
from jadzia.data.transform.base import Element


class ConvertNone(grain.transforms.Map):
    """Convert `None` values to `np.nan` in preparation for batching."""

    def map(self, element: Element) -> Element:
        """
        Apply the transformation.

        Args:
            element: Input element.

        Returns:
            Transformed element.
        """
        return {key: ConvertNone.convert(value) for key, value in element.items()}

    @staticmethod
    def convert[T: Any](value: T | None) -> T | float:
        """Convert `None` to `np.nan`."""
        return value if value is not None else np.nan


class SelectKeys(grain.transforms.Map):
    """Select only specific output key-value pairs."""

    def __init__(self, keys: Iterable[str]):
        """
        Initialize the transformation.

        Args:
            keys: Keys to select from each element.
        """
        self._keys = tuple(keys)

    def map(self, element: Element) -> Element:
        """
        Apply the transformation.

        Args:
            element: Input element.

        Returns:
            Transformed element.
        """
        return {key: element[key] for key in self._keys}


__all__ = ["ConvertNone", "SelectKeys"]
