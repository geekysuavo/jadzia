"""Sequential sampler from [grain.samplers][]."""

# Standard
from typing import Self

# Third party
import polars as pl
from grain.samplers import SequentialSampler


class Sequential(SequentialSampler):
    """Wrapper around [grain.samplers.SequentialSampler][]."""

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        seed: int | None = None,
    ) -> tuple[Self, pl.DataFrame]:
        """
        Build a sequential sampler from a data frame.

        Args:
            df: Input data frame.
            seed: Random seed.

        Returns:
            Newly built sampler instance.
            Potentially updated data frame.
        """
        sampler = cls(num_records=len(df), seed=seed)
        return (sampler, df)


__all__ = ["Sequential"]
