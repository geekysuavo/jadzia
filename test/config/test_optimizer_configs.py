"""Tests of optimizer configurations."""

# Third party
import pytest
from optax import GradientTransformation
from pydantic import TypeAdapter

# Dax
from jadzia.config.optimizer import Optimizer

config_adapter = TypeAdapter(Optimizer)
config_data = [
    {"type": "Adam", "learning_rate": 0.01},
    {"type": "AdamW", "learning_rate": 0.01},
    {"type": "AdamW", "learning_rate": {"type": "Constant", "value": 0.1}},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_optimizer_configs(data):
    """Optimizer configurations validate and build gradient transformations."""
    config = config_adapter.validate_python(data)
    obj = config.build_instance()
    assert isinstance(obj, GradientTransformation)
