"""Tests of sampler configurations."""

# Third party
import polars as pl
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.config.sampler import Sampler
from jadzia.core.validation.context import RuntimeContext

config_adapter = TypeAdapter(Sampler)
config_data = [
    {"type": "General"},
    {"type": "General", "shuffle": True},
    {"type": "General", "shuffle": True, "num_epochs": 2},
    {"type": "Sequential"},
]


@pytest.fixture
def context(tmp_path):
    return RuntimeContext(name="test", path=tmp_path, seed=42)


@pytest.fixture(scope="session")
def df():
    data = {
        "x": [1, 2, 1, 2, 1, 2],
        "y": [3, 3, 4, 4, 5, 5],
    }
    return pl.DataFrame(data)


@pytest.mark.parametrize("data", config_data, ids=range(len(config_data)))
def test_sampler(data, context, df):
    """Sampler configurations validate and build samplers from data frames."""
    config = config_adapter.validate_python(data, context=context)
    (obj, new_df) = config.build_instance(df)
    assert isinstance(obj, config.instance_type)
    assert isinstance(new_df, pl.DataFrame)
