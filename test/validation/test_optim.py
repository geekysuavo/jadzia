"""Tests of validated optimization classes."""

# Standard
from typing import ClassVar

# Third party
import equinox as eqx
import jax.random as jr
import optax

# Dax
from jadzia.core.validation.optim import (
    OptimizerConfig,
    OptimizerInstanceType,
    ScheduleConfig,
    ScheduleInstanceType,
)


class OptimizerCls(OptimizerConfig):
    instance_type: ClassVar[OptimizerInstanceType] = optax.adam
    learning_rate: float = 0.001


class ScheduleCls(ScheduleConfig):
    instance_type: ClassVar[ScheduleInstanceType] = optax.constant_schedule
    value: int = 42


def test_optimizer():
    """Optimizers may be built from their configurations."""
    config = OptimizerCls()
    optimizer = config.build_instance()
    assert isinstance(optimizer, optax.GradientTransformation)

    model = eqx.nn.Linear(8, 1, key=jr.key(42))
    state = config.init_state(optimizer, model)
    assert isinstance(state, tuple)
    assert len(state)


def test_schedule():
    """Schedules may be built from their configurations."""
    config = ScheduleCls()
    schedule = config.build_instance()
    assert schedule(0) == 42
