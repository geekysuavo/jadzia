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
@pytest.mark.parametrize("missing_ref", [False, True])
def test_model_config(missing_ref, unused_block, dag):
    """The composite model configuration validates and builds a module correctly."""
    config_data = {
        "blk_in": {
            "type": "Linear",
            "input": "batch.feats",
            "num_input_features": 3,
            "num_output_features": 16,
        },
        "blk_out": {
            "type": "Linear",
            "input": "blk_in.output",
            "num_input_features": 16,
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
        "blk_extra": {
            "type": "VirtualLinear",
            "input": "blk_out.output",
            "ref": "blk_out",
        },
    }

    outputs = (Identifier("blk_out", "output"), Identifier("blk_cycle_b", "output"))
    key = jr.key(42)

    if not unused_block:
        del config_data["blk_extra"]
    elif missing_ref:
        config_data["blk_extra"]["ref"] = "blk_oops"

    if dag:
        outputs = outputs[:1]
        del config_data["blk_cycle_a"]
        del config_data["blk_cycle_b"]

    if unused_block and missing_ref:
        with pytest.raises(ValueError, match="reference blocks are missing"):
            CompositeModel.model_validate(config_data)
        return

    config = CompositeModel.model_validate(config_data)
    assert config.requires_key is True

    if not dag:
        with pytest.raises(ValueError, match="graph is not a DAG"):
            config.build_instance(key, outputs=outputs)
        return

    with pytest.raises(ValueError, match="requires a key"):
        config.build_instance(outputs=outputs)

    model = config.build_instance(key, outputs=outputs)
    assert isinstance(model, config.instance_type)
    assert "blk_extra" not in model.modules

    (model, state) = config.build_instance(key, outputs=outputs, with_state=True)
    assert isinstance(model, config.instance_type)
    assert isinstance(state, eqx.nn.State)


def test_model_without_key():
    """The composite model only accepts a key when its modules require one."""
    config = CompositeModel(
        split={
            "type": "UnstackFission",
            "input": "batch.data",
            "outputs": ["first", "second"],
            "axis": -1,
        },
    )

    key = jr.key(42)
    outputs = (Identifier("split", "first"),)

    with pytest.raises(ValueError, match="does not require a key"):
        config.build_instance(key, outputs=outputs)

    model = config.build_instance(outputs=outputs)
    assert isinstance(model, config.instance_type)

    (model, state) = config.build_instance(outputs=outputs, with_state=True)
    assert isinstance(model, config.instance_type)
    assert isinstance(state, eqx.nn.State)


def test_model_with_virtual():
    """Virtual blocks are not built into modules."""
    config = CompositeModel(
        query={
            "type": "Linear",
            "input": "a.x",
            "num_input_features": 8,
            "num_output_features": 1,
        },
        copy={"type": "VirtualLinear", "input": "a.z", "ref": "query"},
        fusion={"type": "SumFusion", "inputs": ["query.output", "copy.output"]},
    )

    key = jr.key(42)
    outputs = (Identifier("fusion", "output"),)
    model = config.build_instance(key, outputs=outputs)
    assert "copy" not in model.modules
    assert sum(block.name == "copy" for block in model.blocks) == 0
    assert sum(block.name == "query" for block in model.blocks) == 2
