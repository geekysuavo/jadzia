"""Model building block configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
import equinox as eqx
from pydantic import Field, PositiveInt, Tag

# Dax
import jadzia.nn.fission as fission
import jadzia.nn.fusion as fusion
from jadzia.core.validation.block import FissionBlock, FusionBlock
from jadzia.core.validation.module import ModuleSignature
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class SplitFission(FissionBlock):
    """Configuration for [jadzia.nn.fission.SplitFission][]."""

    instance_type: ClassVar[type[eqx.Module]] = fission.SplitFission
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        fission.SplitFission
    )

    axis: int
    sizes: Annotated[tuple[PositiveInt, ...], Field(min_length=2)]


class UnstackFission(FissionBlock):
    """Configuration for [jadzia.nn.fission.UnstackFission][]."""

    instance_type: ClassVar[type[eqx.Module]] = fission.UnstackFission
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        fission.UnstackFission
    )

    axis: int


class MeanFusion(FusionBlock):
    """Configuration for [jadzia.nn.fusion.MeanFusion][]."""

    instance_type: ClassVar[type[eqx.Module]] = fusion.MeanFusion
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        fusion.MeanFusion
    )


class MaxFusion(FusionBlock):
    """Configuration for [jadzia.nn.fusion.MaxFusion][]."""

    instance_type: ClassVar[type[eqx.Module]] = fusion.MaxFusion
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        fusion.MaxFusion
    )


class SumFusion(FusionBlock):
    """Configuration for [jadzia.nn.fusion.SumFusion][]."""

    instance_type: ClassVar[type[eqx.Module]] = fusion.SumFusion
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        fusion.SumFusion
    )


type Block = Annotated[
    Annotated[SplitFission, Tag("SplitFission")]
    | Annotated[UnstackFission, Tag("UnstackFission")]
    | Annotated[SumFusion, Tag("SumFusion")]
    | Annotated[MeanFusion, Tag("MeanFusion")]
    | Annotated[MaxFusion, Tag("MaxFusion")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported model building block configurations."""

__all__ = ["Block"]
