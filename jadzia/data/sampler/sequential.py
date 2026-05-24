"""Sequential sampler from [grain.samplers][]."""

# Standard
from typing import Self

# Third party
import polars as pl
from grain.samplers import SequentialSampler
from grain.sharding import NoSharding, ShardOptions


class Sequential(SequentialSampler):
    """Wrapper around [grain.samplers.SequentialSampler][]."""

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        seed: int | None = None,
        shard_options: ShardOptions = NoSharding(),
    ) -> tuple[Self, pl.DataFrame]:
        """
        Build a sequential sampler from a data frame.

        Args:
            df: Input data frame.
            seed: Random seed.
            shard_options: Sharding options.

        Returns:
            Newly built sampler instance.
            Potentially updated data frame.
        """
        sampler = cls(num_records=len(df), shard_options=shard_options, seed=seed)
        return (sampler, df)


__all__ = ["Sequential"]
