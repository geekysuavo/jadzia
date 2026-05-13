"""Activation function configurations."""

# Standard
from collections.abc import Callable
from typing import Annotated, Any, ClassVar

# Third party
import jax
from pydantic import BeforeValidator, PositiveFloat, Tag

# Dax
from jadzia.core.validation.function import FunctionConfig
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer

type ActivationType = Callable[..., jax.Array]
"""Activation function type, used by configurations."""


class ReLU(FunctionConfig):
    """Configuration for [jax.nn.relu][]."""

    instance_type: ClassVar[ActivationType] = jax.nn.relu


class LeakyReLU(FunctionConfig):
    """Configuration for [jax.nn.leaky_relu][]."""

    instance_type: ClassVar[ActivationType] = jax.nn.leaky_relu
    negative_slope: PositiveFloat = 0.01


def _activation_from_str(value: Any) -> Any:
    """Promote activation names to dictionaries for use within discriminated unions."""
    if isinstance(value, str):
        return {"type": value}
    else:
        return value


type Activation = Annotated[
    Annotated[ReLU, Tag("ReLU")] | Annotated[LeakyReLU, Tag("LeakyReLU")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported activation function configurations."""

type ActivationStr = Annotated[Activation, BeforeValidator(_activation_from_str)]
"""
Type alias for coercing activation function configurations from their names.

Activation functions may be specified by name,
```yaml
activation: SiLU
```
or with additional arguments when needed,
```yaml
activation:
  type: SELU
  alpha: 0.8
```
"""

__all__ = ["Activation", "ActivationStr"]
