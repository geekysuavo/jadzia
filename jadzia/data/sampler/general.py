"""General index sampler from [grain.samplers][]."""

# Standard
from collections.abc import Sequence
from typing import Self

# Third party
import polars as pl
from grain.samplers import IndexSampler


class General(IndexSampler):
    """Wrapper around [grain.samplers.IndexSampler][]."""

    @classmethod
    def from_dfs(
        cls,
        dfs: Sequence[pl.DataFrame],
        seed: int | None = None,
        shuffle: bool = False,
        num_epochs: int | None = None,
    ) -> tuple[Self, Sequence[pl.DataFrame]]:
        """
        Build an index sampler from data frames.

        Args:
            dfs: Input data frames.
            seed: Random seed.
            shuffle: Whether to globally shuffle the dataset.
            num_epochs: Number of epochs to repeat the dataset,
                or `None` to repeat indefinitely.

        Returns:
            Newly built sampler instance.
            Potentially updated data frames.
        """
        sampler = cls(
            num_records=sum(map(len, dfs)),
            seed=seed,
            shuffle=shuffle,
            num_epochs=num_epochs,
        )
        return (sampler, dfs)


__all__ = ["General"]
