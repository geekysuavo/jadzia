"""Tests of transformation configurations."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.transform import Transform

config_adapter = TypeAdapter(Transform)
config_data = [
    {"type": "Insert", "value": 1},
    {"type": "InsertCopy", "input": "a.b"},
    {"type": "InsertFromSequence", "values": ("x", "y")},
    {"type": "InsertRandomChoice", "values": ("x", "y")},
    {"type": "InsertRandomInteger"},
    {"type": "InsertRandomUniform"},
    {"type": "Unary", "input": "a.b", "function": "math.sin"},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_transform_configs(data):
    """Transformation configurations validate and build callables."""
    config = config_adapter.validate_python(data)
    transform = config.build_instance(name="tfm")
    assert isinstance(transform, config.instance_type)


def test_validators():
    """Transformation configurations validate their fields."""
    invalid_data = {"type": "InsertRandomInteger", "upper": 0}
    with pytest.raises(ValueError, match="lower bound must be less"):
        config_adapter.validate_python(invalid_data)

    invalid_data = {"type": "InsertRandomUniform", "upper": 0}
    with pytest.raises(ValueError, match="lower bound must be less"):
        config_adapter.validate_python(invalid_data)
