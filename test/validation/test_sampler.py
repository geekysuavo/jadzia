"""Tests of validated sampler classes."""

# Standard
from collections.abc import Sequence
from typing import Self

# Third party
import polars as pl
import pytest
from grain import RecordMetadata
from grain.sharding import NoSharding, ShardOptions

# Dax
from jadzia.core.validation.context import RuntimeContext
from jadzia.core.validation.sampler import SamplerConfig

_no_sharding = NoSharding()


class ExampleSampler:
    def __init__(
        self,
        num_records: int,
        seed: int | None,
        key: int,
        shard_options: ShardOptions = _no_sharding,
    ):
        self.key = key

    def __getitem__(self, index: int) -> RecordMetadata:
        return RecordMetadata(index=index, record_key=self.key, rng=None)

    @classmethod
    def from_dfs(
        cls,
        dfs: Sequence[pl.DataFrame],
        seed: int | None,
        key: int,
    ) -> tuple[Self, tuple[pl.DataFrame, ...]]:
        return (cls(sum(map(len, dfs)), seed, key), tuple(dfs))


@pytest.mark.parametrize("key", [2, 3])
def test_sampler(tmp_path, key):
    """Signature discovery of samplers produces the expected results."""
    config_cls = SamplerConfig.auto(ExampleSampler)
    assert issubclass(config_cls, SamplerConfig)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()

    with pytest.raises(ValueError, match="runtime context not available"):
        config_cls(key=key)

    data = {"key": key}
    context = RuntimeContext(name="test", path=tmp_path)
    config = config_cls.model_validate(data, context=context)

    df_a = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df_b = pl.DataFrame({"x": [3, 2], "y": [7, 8]})
    (obj, new_dfs) = config.build_instance([df_a, df_b])
    assert isinstance(obj, ExampleSampler)
    assert isinstance(new_dfs, tuple)
    (new_df_a, new_df_b) = new_dfs
    assert new_df_a is df_a
    assert new_df_b is df_b

    metadata = obj[12]
    assert metadata.index == 12
    assert metadata.record_key == key
    assert metadata.rng is None
