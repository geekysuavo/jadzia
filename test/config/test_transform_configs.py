"""Tests of transformation configurations."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.transform import Transform

config_adapter = TypeAdapter(Transform)
config_data = [
    {"type": "Insert", "inputs": {}, "value": 1},
    {"type": "InsertCopy", "inputs": {"source": "a.b"}},
    {"type": "InsertFromSequence", "inputs": {}, "values": ("x", "y")},
    {"type": "InsertRandomChoice", "inputs": {}, "values": ("x", "y")},
    {"type": "InsertRandomInteger", "inputs": {}},
    {"type": "InsertRandomUniform", "inputs": {}},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_transform_configs(data):
    """Transformation configurations validate and build callables."""
    config = config_adapter.validate_python(data)
    transform = config.build_instance(name="tfm")
    assert isinstance(transform, config.instance_type)


def test_validators():
    """Transformation configurations validate their fields."""
    invalid_data = {"type": "InsertRandomInteger", "inputs": {}, "upper": 0}
    with pytest.raises(ValueError, match="lower bound must be less"):
        config_adapter.validate_python(invalid_data)

    invalid_data = {"type": "InsertRandomUniform", "inputs": {}, "upper": 0}
    with pytest.raises(ValueError, match="lower bound must be less"):
        config_adapter.validate_python(invalid_data)
