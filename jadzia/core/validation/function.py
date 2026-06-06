"""Validation base class for functions."""

# Standard
import functools
import math
from collections.abc import Callable
from typing import Annotated, Any, ClassVar

# Third party
import jax
import numpy as np
from pydantic import BeforeValidator, PlainSerializer

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


def _validate_function_str(value: Any) -> Callable:
    """Resolve a function name to its associated callable."""
    if not isinstance(value, str):
        raise TypeError(f"expected str, got {type(value).__name__}")

    (module, *attrs) = value.split(".")
    if module == "math":
        obj = math
    elif module == "jax":
        obj = jax
    elif module in ("np", "numpy"):
        obj = np
    else:
        raise ValueError(f"function '{value}' is not supported")

    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            raise ValueError(f"attribute '{attr}' not found in {obj.__name__}")

    if callable(obj):
        return obj
    else:
        raise ValueError(f"object '{value}' is not callable")


def _serialize_function_str(func: Any) -> str:
    """Convert a function to its name."""
    (module, name) = (func.__module__, func.__name__)
    if any(module.startswith(prefix) for prefix in ("math", "jax", "np", "numpy")):
        return f"{module}.{name}"
    else:
        raise ValueError(f"function '{func}' is not supported")


type FunctionStr = Annotated[
    Callable,
    BeforeValidator(_validate_function_str),
    PlainSerializer(_serialize_function_str),
]
"""Function that is convertible to and from its string representation."""

__all__ = ["FunctionConfig", "FunctionStr"]
