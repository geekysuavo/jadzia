"""Transformation base classes."""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, overload

# Third party
import grain
from numpy.random import Generator

type Element = dict[str, Any]
"""Expected type of each transformed element."""


@dataclass(frozen=True)
class _Base(ABC):
    """
    Transformation base class.

    Attributes:
        input_names: Input example keys.
        output_names: Output example keys.
    """

    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Transformation implementation."""

    @overload
    def _call_impl(
        self,
        element: Element,
        *,
        as_bool: Literal[False] = False,
        **kwargs: int | Generator,
    ) -> Element: ...

    @overload
    def _call_impl(
        self,
        element: Element,
        *,
        as_bool: Literal[True] = True,
    ) -> bool: ...

    def _call_impl(
        self,
        element: Element,
        *,
        as_bool: bool = False,
        **kwargs: int | Generator,
    ) -> Element | bool:
        """
        Route element values to and from transformation implementation calls.

        Args:
            element: Input element.
            as_bool: Whether to return a boolean value.
            kwargs: Additional keyword arguments passed to the implementation.

        Returns:
            Transformed element, or a boolean value if `as_bool` is true.
        """
        call_inputs = (element[name] for name in self.input_names)
        call_outputs = self(*call_inputs, **kwargs)
        if as_bool:
            return bool(call_outputs)

        if len(self.output_names) == 1:
            call_outputs = (call_outputs,)

        updates = dict(zip(self.output_names, call_outputs, strict=True))
        element_out = element.copy()
        element_out.update(updates)
        return element_out


@dataclass(frozen=True)
class Map(_Base, grain.transforms.Map):
    """Map transformation base class."""

    def map(self, element: Element) -> Element:
        """
        Apply the transformation.

        Args:
            element: Input element.

        Returns:
            Transformed element.
        """
        return self._call_impl(element)


@dataclass(frozen=True)
class MapWithIndex(_Base, grain.transforms.MapWithIndex):
    """Indexed map transformation base class."""

    def map_with_index(self, index: int, element: Element) -> Element:
        """
        Apply the transformation.

        Args:
            index: Element index.
            element: Input element.

        Returns:
            Transformed element.
        """
        return self._call_impl(element, index=index)


@dataclass(frozen=True)
class RandomMap(_Base, grain.transforms.RandomMap):
    """Random map transformation base class."""

    def random_map(self, element: Element, rng: Generator) -> Element:
        """
        Apply the transformation.

        Args:
            element: Input element.
            rng: Random number generator.

        Returns:
            Transformed element.
        """
        return self._call_impl(element, rng=rng)


@dataclass(frozen=True)
class NumpyRandomMap(_Base, grain.transforms.TfRandomMap):
    """Random [numpy][] map transformation base class."""

    def np_random_map(self, element: Element, rng: Generator) -> Element:
        """
        Apply the transformation.

        Args:
            element: Input element.
            rng: Random number generator.

        Returns:
            Transformed element.
        """
        return self._call_impl(element, rng=rng)


@dataclass(frozen=True)
class Filter(_Base, grain.transforms.Filter):
    """Filter transformation base class."""

    def filter(self, element: Element) -> bool:
        """
        Apply the transformation.

        Args:
            element: Input element.

        Returns:
            Whether the element should be retained.
        """
        return self._call_impl(element, as_bool=True)


__all__ = ["Filter", "Map", "MapWithIndex", "NumpyRandomMap", "RandomMap"]
