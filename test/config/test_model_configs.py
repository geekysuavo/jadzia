"""Tests of model configurations."""

# Third party
import equinox as eqx
import jax.random as jr
import pytest

# Dax
from jadzia.config.model import CompositeModel
from jadzia.core.dependency.identifier import Identifier


@pytest.mark.parametrize("dag", [True, False])
@pytest.mark.parametrize("unused_block", [False, True])
@pytest.mark.parametrize("missing_input", [False, True])
def test_model_config(missing_input, unused_block, dag):
    """The composite model configuration validates and builds a module correctly."""
    config_data = {
        "outputs": {
            "yhat": "blk_out.output",
            "cycle": "blk_cycle_b.output",
        },
        "blocks": {
            "blk_in": {
                "type": "Linear",
                "input": Identifier.from_str("batch.feats", is_batch=True),
                "num_input_features": 3,
                "num_output_features": 16,
            },
            "blk_out": {
                "type": "Linear",
                "input": "blk_in.output",
                "num_input_features": 16,
                "num_output_features": 1,
            },
            "blk_extra": {
                "type": "Linear",
                "input": "blk_out.output",
                "num_input_features": 1,
                "num_output_features": 1,
            },
            "blk_cycle_a": {
                "type": "Linear",
                "input": "blk_cycle_b.output",
                "num_input_features": 1,
                "num_output_features": 1,
            },
            "blk_cycle_b": {
                "type": "Linear",
                "input": "blk_cycle_a.output",
                "num_input_features": 1,
                "num_output_features": 1,
            },
        },
    }

    if missing_input:
        config_data["blocks"]["blk_out"]["input"] = "blk_oops.output"

    if not unused_block:
        del config_data["blocks"]["blk_extra"]

    if dag:
        del config_data["outputs"]["cycle"]
        del config_data["blocks"]["blk_cycle_a"]
        del config_data["blocks"]["blk_cycle_b"]

    if missing_input:
        with pytest.raises(ValueError, match="block inputs are not satisfied"):
            CompositeModel.model_validate(config_data)
        return

    if unused_block:
        with pytest.raises(ValueError, match="blocks are unused"):
            CompositeModel.model_validate(config_data)
        return

    if not dag:
        with pytest.raises(ValueError, match="dependencies do not form a DAG"):
            CompositeModel.model_validate(config_data)
        return

    config = CompositeModel.model_validate(config_data)
    assert config.input_names == ("feats",)
    assert config.output_names == ("yhat",)

    with pytest.raises(ValueError, match="requires a key"):
        config.build_instance()

    key = jr.key(42)
    model = config.build_instance(key)
    assert isinstance(model, config.instance_type)

    (model, state) = config.build_instance(key, with_state=True)
    assert isinstance(model, config.instance_type)
    assert isinstance(state, eqx.nn.State)


def test_model_without_key():
    """The composite model only accepts a key when its modules require one."""
    config = CompositeModel(
        outputs={"yhat": "split.first", "zhat": "split.second"},
        blocks={
            "split": {
                "type": "UnstackFission",
                "input": Identifier.from_str("batch.data", is_batch=True),
                "outputs": ["first", "second"],
                "axis": -1,
            },
        },
    )

    key = jr.key(42)
    with pytest.raises(ValueError, match="does not require a key"):
        config.build_instance(key)

    model = config.build_instance()
    assert isinstance(model, config.instance_type)

    (model, state) = config.build_instance(with_state=True)
    assert isinstance(model, config.instance_type)
    assert isinstance(state, eqx.nn.State)
