"""Validation base classes for samplers."""

# Standard
from typing import Any, ClassVar, Protocol, Self

# Third party
import polars as pl
from grain.samplers import Sampler as GrainSampler

# Dax
from jadzia.core.validation.buildable import BuildableConfig
from jadzia.core.validation.context import WithRuntimeContext


class Sampler(GrainSampler, Protocol):
    """A [grain.samplers.Sampler][] that may be built from a data frame."""

    @classmethod
    def from_df(
        cls,
        df: pl.DataFrame,
        seed: int | None,
        **kwargs: Any,
    ) -> tuple[Self, pl.DataFrame]: ...


type SamplerInstanceType = type[Sampler]
"""Type of all sampler instances."""


class SamplerConfig(BuildableConfig[Sampler], WithRuntimeContext):
    """Sampler configuration base class."""

    instance_type: ClassVar[SamplerInstanceType]

    def build_instance(  # ty: ignore[invalid-method-override]
        self,
        df: pl.DataFrame,
    ) -> tuple[Sampler, pl.DataFrame]:
        """
        Build an instance based on this configuration.

        This method returns an updated data frame, which may or may not be the
        same data frame that was passed as input. This is done to allow samplers
        an opportunity to reorganize data frame rows as needed.

        Args:
            df: Input data frame.

        Returns:
            Newly built sampler instance.
            Potentially updated data frame.
        """
        cls = type(self)
        seed = self._context.seed
        kwargs = self.model_dump_for_instance()
        return cls.instance_type.from_df(df, seed, **kwargs)

    @classmethod
    def auto_hook(
        cls,
        instance_type: SamplerInstanceType,
        *,
        field_defs: dict[str, tuple[Any, Any]],
    ):
        """Modify the field definitions obtained by inspection."""
        # remove fields provided by this class.
        field_defs.pop("num_records")
        field_defs.pop("shard_options")
        field_defs.pop("seed")


__all__ = ["SamplerConfig", "SamplerInstanceType"]
