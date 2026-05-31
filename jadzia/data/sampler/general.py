"""General index sampler from [grain.samplers][]."""

# Standard
from typing import Self

# Third party
import polars as pl
from grain.samplers import IndexSampler


class General(IndexSampler):
    """Wrapper around [grain.samplers.IndexSampler][]."""

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        seed: int | None = None,
        shuffle: bool = False,
        num_epochs: int | None = None,
    ) -> tuple[Self, pl.DataFrame]:
        """
        Build an index sampler from a data frame.

        Args:
            df: Input data frame.
            seed: Random seed.
            shuffle: Whether to globally shuffle the dataset.
            num_epochs: Number of epochs to repeat the dataset,
                or `None` to repeat indefinitely.

        Returns:
            Newly built sampler instance.
            Potentially updated data frame.
        """
        sampler = cls(
            num_records=len(df),
            seed=seed,
            shuffle=shuffle,
            num_epochs=num_epochs,
        )
        return (sampler, df)


__all__ = ["General"]
