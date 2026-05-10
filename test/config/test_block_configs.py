"""Tests of model building block configurations."""

# Third party
import jax.random as jr
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.block import Block

config_adapter = TypeAdapter(Block)
config_data = [
    {"type": "MaxFusion", "inputs": ["a.b", "c.d"]},
    {"type": "MeanFusion", "inputs": ["a.b", "c.d"]},
    {"type": "SumFusion", "inputs": ["a.b", "c.d"]},
    {
        "type": "SplitFission",
        "input": "a.b",
        "outputs": ["two", "three"],
        "axis": -1,
        "sizes": [2, 3],
    },
    {"type": "UnstackFission", "input": "a.b", "num_outputs": 5, "axis": 0},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_block_configs(data):
    """Model building block configurations validate and build loss modules."""
    config = config_adapter.validate_python(data)
    key = jr.key(42) if config.requires_key else None
    obj = config.build_instance(key)
    assert isinstance(obj, config.instance_type)
