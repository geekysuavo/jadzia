"""Tests of loss function configurations."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.loss.composite import CompositeLoss
from jadzia.config.loss.function import LossFunction

config_adapter = TypeAdapter(LossFunction)
config_data = [
    {"type": "Direct", "inputs": {"prediction": "in"}},
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


def test_composite_loss_config():
    """The composite loss configuration validates and builds a module correctly."""
    config = CompositeLoss(
        terms={"mse": 0.5, "huber": 0.125, "direct": 1.0},
        functions={
            "mse": {"type": "MSE", "input": "foo"},
            "huber": {"type": "Huber", "input": "bar", "delta": 0.1},
            "direct": {"type": "Direct", "inputs": {"prediction": "zoo"}},
        },
    )
    assert config.prediction_names == ("bar", "foo", "zoo")
    assert config.reference_names == ("bar", "foo")

    obj = config.build_instance()
    assert isinstance(obj, config.instance_type)

    with pytest.raises(ValueError, match="invalid composite loss"):
        CompositeLoss(terms={"mse": 0.5}, functions={})

    with pytest.raises(ValueError, match="invalid composite loss"):
        CompositeLoss(
            terms={"mse": 0.5},
            functions={
                "mse": {"type": "MSE", "input": "foo"},
                "mae": {"type": "MAE", "input": "bar"},
            },
        )
