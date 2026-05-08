"""Tests of schedule configurations."""

# Third party
import jax
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.schedule import Schedule

config_adapter = TypeAdapter(Schedule)
config_data = [
    {"type": "Constant", "value": 0.01},
    {"type": "CosineDecay", "init_value": 0.001, "decay_steps": 10},
]

@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_schedule_configs(data):
    """Schedule configurations validate and build callables."""
    config = config_adapter.validate_python(data)
    obj = config.build_instance()
    assert isinstance(obj(0), float | int | jax.Array)
