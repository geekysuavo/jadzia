"""Schedule configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
import optax
from pydantic import PositiveFloat, PositiveInt, Tag

# Dax
from jadzia.core.validation.interval import UnitInterval
from jadzia.core.validation.optim import ScheduleConfig, ScheduleInstanceType
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class Constant(ScheduleConfig):
    """
    Configuration for [optax.schedules.constant_schedule][].

    Attributes:
        value: Value to be held constant.
    """

    instance_type: ClassVar[ScheduleInstanceType] = optax.constant_schedule

    value: float


class CosineDecay(ScheduleConfig):
    r"""
    Configuration for [optax.schedules.cosine_decay_schedule][].

    Attributes:
        init_value: $I$, Initial value of the learning rate.
        decay_steps: $T$, Number of steps for which to apply the decay.
        alpha: $\alpha$, Multiplier value used to adjust the learning rate.
        exponent: $p$, Value modifying the shape of the cosine factor.
    """

    instance_type: ClassVar[ScheduleInstanceType] = optax.cosine_decay_schedule

    init_value: float
    decay_steps: PositiveInt
    alpha: UnitInterval = 0.0
    exponent: PositiveFloat = 1.0


type Schedule = Annotated[
    Annotated[Constant, Tag("Constant")]
    | Annotated[CosineDecay, Tag("CosineDecay")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported schedule configurations."""

__all__ = ["Schedule"]
