"""Sequential sampler from [grain.samplers][]."""

# Standard
from collections.abc import Sequence
from typing import Self

# Third party
import polars as pl
from grain.samplers import SequentialSampler


class Sequential(SequentialSampler):
    """Wrapper around [grain.samplers.SequentialSampler][]."""

    @classmethod
    def from_dfs(
        cls,
        dfs: Sequence[pl.DataFrame],
        seed: int | None = None,
    ) -> tuple[Self, Sequence[pl.DataFrame]]:
        """
        Build a sequential sampler from data frames.

        Args:
            dfs: Input data frames.
            seed: Random seed.

        Returns:
            Newly built sampler instance.
            Potentially updated data frames.
        """
        sampler = cls(num_records=sum(map(len, dfs)), seed=seed)
        return (sampler, dfs)


__all__ = ["Sequential"]
