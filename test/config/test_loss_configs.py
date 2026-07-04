"""Tests of loss function configurations."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.loss.composite import CompositeLoss
from jadzia.config.loss.function import LossFunction

config_adapter = TypeAdapter(LossFunction)
config_data = [
    {"type": "Direct", "inputs": {"prediction": "a.b"}},
    {"type": "MAE", "inputs": {"yhat": "a.b", "y": "a.b"}},
    {"type": "MSE", "inputs": {"yhat": "a.b", "y": "a.b"}},
    {"type": "Huber", "inputs": {"yhat": "a.b", "y": "a.b"}},
    {"type": "Huber", "inputs": {"yhat": "a.b", "y": "a.b"}, "delta": 2.0},
]


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_loss_function_configs(data):
    """Loss function configurations validate and build loss modules."""
    config = config_adapter.validate_python(data)
    obj = config.build_instance()
    assert isinstance(obj, config.instance_type)


def test_composite_loss_config(runtime_context):
    """The composite loss configuration validates and builds a module correctly."""
    config_data = {
        "mse": {"type": "MSE", "inputs": {"yhat": "foo.yhat", "y": "foo.y"}},
        "huber": {"type": "Huber", "inputs": {"yhat": "bar.yhat", "y": "bar.y"}},
        "direct": {"type": "Direct", "inputs": {"prediction": "zoo.yhat"}},
    }
    config = CompositeLoss.model_validate(config_data, context=runtime_context)

    (graph, node) = config.build_subgraph()
    assert node in graph
    assert node.is_gather is False
    assert set(graph.sort_nodes(subset=(node,))) == set(graph)

    obj = config.build_instance()
    assert isinstance(obj, config.instance_type)
