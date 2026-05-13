"""Validation base class for functions."""

# Standard
import functools
from collections.abc import Callable
from typing import ClassVar

# Dax
from jadzia.core.validation.buildable import BuildableConfig


class FunctionConfig(BuildableConfig[Callable]):
    """Base class for all validated configurations of functions."""

    instance_type: ClassVar[Callable]

    def build_instance(self) -> Callable:
        """Bind parameters to a function based on this configuration."""
        cls = type(self)
        kwargs = self.model_dump_for_instance()
        return functools.partial(cls.instance_type, **kwargs)


__all__ = ["FunctionConfig"]
