"""Model building block configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
import equinox as eqx
from pydantic import Field, PositiveInt, Tag

# Dax
import jadzia.nn.fission as fission
import jadzia.nn.fusion as fusion
import jadzia.nn.linear as linear
from jadzia.core.validation.block import FissionBlock, FusionBlock, SimpleBlock, Virtual
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


class Linear(SimpleBlock):
    """Configuration for [jadzia.nn.linear.Linear][]."""

    instance_type: ClassVar[type[eqx.Module]] = linear.Linear
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(linear.Linear)
    requires_key: ClassVar[bool] = True

    num_input_features: PositiveInt
    num_output_features: PositiveInt
    bias: bool = True


class VirtualLinear(SimpleBlock, Virtual):
    """Configuration for virtual [jadzia.config.block.Linear][] blocks."""

    instance_type: ClassVar[type[eqx.Module]] = linear.Linear
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(linear.Linear)


type Block = Annotated[
    Annotated[SplitFission, Tag("SplitFission")]
    | Annotated[UnstackFission, Tag("UnstackFission")]
    | Annotated[SumFusion, Tag("SumFusion")]
    | Annotated[MeanFusion, Tag("MeanFusion")]
    | Annotated[MaxFusion, Tag("MaxFusion")]
    | Annotated[Linear, Tag("Linear")]
    | Annotated[VirtualLinear, Tag("VirtualLinear")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported model building block configurations."""

__all__ = ["Block"]
