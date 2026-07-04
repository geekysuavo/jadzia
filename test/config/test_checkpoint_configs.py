"""Tests of checkpointer configurations."""

# Third party
import pytest
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.config.checkpoint import CheckpointConfig


@pytest.mark.parametrize("on_start", [False, True])
@pytest.mark.parametrize("on_preemption", [False, True])
@pytest.mark.parametrize("on_steps", [None, (1, 10, 20)], ids=[None, 3])
@pytest.mark.parametrize("retain", [None, 3])
@pytest.mark.parametrize("every", [None, 5])
def test_checkpoint_config(
    runtime_context,
    every,
    retain,
    on_steps,
    on_preemption,
    on_start,
):
    """The checkpoint configuration builds checkpointers correctly."""
    config_data = {
        "every": every,
        "retain": retain,
        "on_steps": on_steps,
        "on_preemption": on_preemption,
        "on_start": on_start,
    }

    config = CheckpointConfig.model_validate(config_data, context=runtime_context)

    checkpointer = config.build_instance()
    assert isinstance(checkpointer, ocp.training.Checkpointer)

    for step in range(2):
        checkpointer.save(step, {"step": step})
