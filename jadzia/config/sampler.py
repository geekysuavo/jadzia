"""Sampler configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
from pydantic import Tag

# Dax
import jadzia.data.sampler.general as general
import jadzia.data.sampler.sequential as sequential
from jadzia.core.validation.sampler import SamplerConfig, SamplerInstanceType
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class General(SamplerConfig):
    """
    Configuration for [grain.samplers.IndexSampler][].

    Attributes:
        shuffle: Whether to globally shuffle the dataset.
        num_epochs: Number of epochs to repeat the dataset, or `None` for indefinite.
    """

    instance_type: ClassVar[SamplerInstanceType] = general.General

    shuffle: bool = False
    num_epochs: int | None = None


class Sequential(SamplerConfig):
    """Configuration for [grain.samplers.SequentialSampler][]."""

    instance_type: ClassVar[SamplerInstanceType] = sequential.Sequential


type Sampler = Annotated[
    Annotated[General, Tag("General")]
    | Annotated[Sequential, Tag("Sequential")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported sampler configurations."""

__all__ = ["Sampler"]
