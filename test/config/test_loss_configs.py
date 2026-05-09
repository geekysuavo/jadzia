"""Tests of loss function configurations."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.loss import LossFunction

config_adapter = TypeAdapter(LossFunction)
config_data = [
    {"type": "MAE", "input": "in"},
    {"type": "MSE", "input": "in"},
    {"type": "Huber", "input": "in"},
    {"type": "Huber", "input": "in", "delta": 2.0},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_loss_function_configs(data):
    """Loss function configurations validate and build loss modules."""
    config = config_adapter.validate_python(data)
    obj = config.build_instance()
    assert isinstance(obj, config.instance_type)
