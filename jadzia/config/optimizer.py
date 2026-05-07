"""Optimizer configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
import optax
from pydantic import NonNegativeFloat, PositiveFloat, Tag

# Dax
from jadzia.core.validation.interval import UnitIntervalOpen
from jadzia.core.validation.optimizer import OptimizerConfig, OptimizerInstanceType
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class Adam(OptimizerConfig):
    r"""
    Configuration for [optax.adam][].

    Attributes:
        learning_rate: $\alpha_t$, Learning rate or learning rate schedule.
        b1: $\beta_1$, First moment decay rate parameter.
        b2: $\beta_2$, Second moment decay rate parameter.
        eps: $\varepsilon$, Small constant outside the square root.
        eps_root: $\tilde\varepsilon$, Small constant inside the square root.
        nesterov: Whether to use Nesterov momentum.
    """

    instance_type: ClassVar[OptimizerInstanceType] = optax.adam

    learning_rate: PositiveFloat
    b1: UnitIntervalOpen = 0.9
    b2: UnitIntervalOpen = 0.999
    eps: PositiveFloat = 1.0e-8
    eps_root: NonNegativeFloat = 0.0
    nesterov: bool = False


class AdamW(OptimizerConfig):
    r"""
    Configuration for [optax.adamw][].

    Attributes:
        learning_rate: $\alpha_t$, Learning rate or learning rate schedule.
        b1: $\beta_1$, First moment decay rate parameter.
        b2: $\beta_2$, Second moment decay rate parameter.
        eps: $\varepsilon$, Small constant outside the square root.
        eps_root: $\tilde\varepsilon$, Small constant inside the square root.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: $\lambda$, Strength of weight decay regularization.
    """

    instance_type: ClassVar[OptimizerInstanceType] = optax.adamw

    learning_rate: PositiveFloat
    b1: UnitIntervalOpen = 0.9
    b2: UnitIntervalOpen = 0.999
    eps: PositiveFloat = 1.0e-8
    eps_root: NonNegativeFloat = 0.0
    nesterov: bool = False
    weight_decay: PositiveFloat = 0.0001


type Optimizer = Annotated[
    Annotated[Adam, Tag("Adam")]
    | Annotated[AdamW, Tag("AdamW")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported optimizer configurations."""

__all__ = ["Optimizer"]
