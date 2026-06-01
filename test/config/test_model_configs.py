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
    }

    outputs = (Identifier("blk_out", "output"), Identifier("blk_cycle_b", "output"))
    key = jr.key(42)

    if missing_input:
        config_data["blk_out"]["input"] = "blk_oops.output"

    if not unused_block:
        del config_data["blk_extra"]

    if dag:
        outputs = outputs[:1]
        del config_data["blk_cycle_a"]
        del config_data["blk_cycle_b"]

    if missing_input:
        with pytest.raises(ValueError, match="block inputs are not satisfied"):
            CompositeModel.model_validate(config_data)
        return

    if not dag:
        with pytest.raises(ValueError, match="dependencies do not form a DAG"):
            CompositeModel.model_validate(config_data)
        return

    config = CompositeModel.model_validate(config_data)
    assert config.input_names == (Identifier("batch", "feats", is_batch=True),)
    assert set(config.output_names) >= set(outputs)

    if unused_block:
        with pytest.raises(ValueError, match="blocks are unused"):
            config.build_instance(key, outputs=outputs)
        return

    with pytest.raises(ValueError, match="missing outputs"):
        config.build_instance(key, outputs=(Identifier("invalid", "id"),))

    with pytest.raises(ValueError, match="requires a key"):
        config.build_instance()

    model = config.build_instance(key, outputs=outputs)
    assert isinstance(model, config.instance_type)

    (model, state) = config.build_instance(key, outputs=outputs, with_state=True)
    assert isinstance(model, config.instance_type)
    assert isinstance(state, eqx.nn.State)


def test_model_without_key():
    """The composite model only accepts a key when its modules require one."""
    config = CompositeModel(
        split={
            "type": "UnstackFission",
            "input": Identifier.from_str("batch.data", is_batch=True),
            "outputs": ["first", "second"],
            "axis": -1,
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
