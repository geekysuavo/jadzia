"""Tests of optimizer configurations."""

# Third party
from optax import GradientTransformation
from pydantic import RootModel

# Dax
from jadzia.config.optimizer import Optimizer


def test_optimizer_configs():
    """Optimizer configurations validate and build gradient transformations."""
    config_data = [
        {"type": "Adam", "learning_rate": 0.01},
        {"type": "AdamW", "learning_rate": 0.01},
    ]

    config_cls = RootModel[list[Optimizer]]
    configs = config_cls.model_validate(config_data)
    for config in configs.root:
        obj = config.build_instance()
        assert isinstance(obj, GradientTransformation)
