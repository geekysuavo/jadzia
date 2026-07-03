"""Tests of validation context."""

# Third party
import pytest

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.context import RuntimeContext, WithRuntimeContext


class ConfigWithContext(Config, WithRuntimeContext):
    x: int = 1


def test_context_provided(tmp_path):
    """Context is available when provided."""
    context = RuntimeContext(name="test", path=tmp_path)
    config = ConfigWithContext.model_validate({}, context=context)
    assert config.context is context


def test_context_not_provided():
    """Lack of runtime context raises an exception during validation."""
    with pytest.raises(ValueError, match="not available"):
        ConfigWithContext()


def test_checkpoint_dir(tmp_path):
    """Context has a `checkpoint_dir` attribute."""
    context = RuntimeContext(name="test", path=tmp_path)
    assert context.checkpoint_dir.exists() is False


def test_device_mesh(tmp_path):
    """Context has a `device_mesh` attribute."""
    context = RuntimeContext(name="test", path=tmp_path)
    assert context.device_mesh.axis_names == ("device",)


def test_data_sharding(tmp_path):
    """Context has a `data_sharding` attribute."""
    context = RuntimeContext(name="test", path=tmp_path)
    assert context.data_sharding.replicated_axes == frozenset()


def test_model_sharding(tmp_path):
    """Context has a `model_sharding` attribute."""
    context = RuntimeContext(name="test", path=tmp_path)
    assert context.model_sharding.replicated_axes == frozenset({"device"})


@pytest.mark.parametrize("seed", [None, 528491])
def test_context_from_dir(tmp_path, seed):
    """Context may be attached to a directory."""
    original = RuntimeContext.from_dir(tmp_path, seed)
    reloaded = RuntimeContext.from_dir(tmp_path)
    if seed:
        assert reloaded == original
    else:
        assert reloaded != original

    with pytest.raises(ValueError, match="seed does not match"):
        RuntimeContext.from_dir(tmp_path, seed=42)
