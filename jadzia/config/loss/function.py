"""Loss function configurations."""

# Standard
from typing import Annotated, ClassVar

# Third party
import equinox as eqx
from pydantic import PositiveFloat, Tag

# Dax
import jadzia.nn.loss.direct as dl
import jadzia.nn.loss.regression as rl
from jadzia.core.validation.loss import GeneralLoss, SimpleLoss
from jadzia.core.validation.module import ModuleSignature
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer


class Direct(GeneralLoss):
    """Configuration for [jadzia.nn.loss.direct.Direct][]."""

    instance_type: ClassVar[type[eqx.Module]] = dl.Direct
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(dl.Direct)


class MAE(SimpleLoss):
    """Configuration for [jadzia.nn.loss.regression.MAE][]."""

    instance_type: ClassVar[type[eqx.Module]] = rl.MAE
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(rl.MAE)


class MSE(SimpleLoss):
    """Configuration for [jadzia.nn.loss.regression.MSE][]."""

    instance_type: ClassVar[type[eqx.Module]] = rl.MSE
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(rl.MSE)


class Huber(SimpleLoss):
    """
    Configuration for [jadzia.nn.loss.regression.Huber][].

    Attributes:
        delta: Threshold between the quadratic and linear loss regions.
    """

    instance_type: ClassVar[type[eqx.Module]] = rl.Huber
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(rl.Huber)

    delta: PositiveFloat = 1.0


type LossFunction = Annotated[
    Annotated[Direct, Tag("Direct")]
    | Annotated[MAE, Tag("MAE")]
    | Annotated[MSE, Tag("MSE")]
    | Annotated[Huber, Tag("Huber")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported loss function configurations."""

__all__ = ["LossFunction"]
