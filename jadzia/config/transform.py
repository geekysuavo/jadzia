"""Transformation configurations."""

# Standard
import sys
from typing import Annotated, ClassVar, Self

# Third party
from pydantic import (
    Field,
    PositiveFloat,
    StrictBool,
    StrictFloat,
    StrictInt,
    Tag,
    model_validator,
)

# Dax
import jadzia.data.transform.auxiliary as aux
import jadzia.data.transform.general as gen
from jadzia.core.validation.buildable import Passthrough
from jadzia.core.validation.function import FunctionStr
from jadzia.core.validation.transform import TransformConfig, TransformSignature
from jadzia.core.validation.union import UnionDiscriminator, UnionSerializer
from jadzia.data.transform.base import Transformation


class Insert(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.Insert][]."""

    instance_type: ClassVar[type[Transformation]] = aux.Insert
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.Insert
    )

    value: StrictBool | StrictFloat | StrictInt | str


class InsertCopy(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.InsertCopy][]."""

    instance_type: ClassVar[type[Transformation]] = aux.InsertCopy
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.InsertCopy
    )


class InsertFromSequence(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.InsertFromSequence][]."""

    instance_type: ClassVar[type[Transformation]] = aux.InsertFromSequence
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.InsertFromSequence
    )

    values: tuple[StrictBool | StrictFloat | StrictInt | str, ...]


class InsertRandomChoice(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.InsertRandomChoice][]."""

    instance_type: ClassVar[type[Transformation]] = aux.InsertRandomChoice
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.InsertRandomChoice
    )

    values: tuple[StrictBool | StrictFloat | StrictInt | str, ...]
    weights: tuple[PositiveFloat, ...] | None = None


class InsertRandomInteger(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.InsertRandomInteger][]."""

    instance_type: ClassVar[type[Transformation]] = aux.InsertRandomInteger
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.InsertRandomInteger
    )

    lower: int = 0
    upper: int = sys.maxsize

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Ensure the lower and upper bounds are valid."""
        if self.lower < self.upper:
            return self
        else:
            raise ValueError("lower bound must be less than upper bound")


class InsertRandomUniform(TransformConfig):
    """Configuration for [jadzia.data.transform.auxiliary.InsertRandomUniform][]."""

    instance_type: ClassVar[type[Transformation]] = aux.InsertRandomUniform
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        aux.InsertRandomUniform
    )

    lower: float = 0
    upper: float = sys.maxsize

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Ensure the lower and upper bounds are valid."""
        if self.lower < self.upper:
            return self
        else:
            raise ValueError("lower bound must be less than upper bound")


class Unary(TransformConfig):
    """Configuration for [jadzia.data.transform.general.Unary][]."""

    instance_type: ClassVar[type[Transformation]] = gen.Unary
    instance_call: ClassVar[TransformSignature] = TransformSignature.from_type(
        gen.Unary
    )

    function: Annotated[FunctionStr, Passthrough]
    args: Annotated[tuple[gen.Arg, ...], Field(default_factory=tuple)]
    kwargs: Annotated[dict[str, gen.Arg], Field(default_factory=dict)]


type Transform = Annotated[
    Annotated[Insert, Tag("Insert")]
    | Annotated[InsertCopy, Tag("InsertCopy")]
    | Annotated[InsertFromSequence, Tag("InsertFromSequence")]
    | Annotated[InsertRandomChoice, Tag("InsertRandomChoice")]
    | Annotated[InsertRandomInteger, Tag("InsertRandomInteger")]
    | Annotated[InsertRandomUniform, Tag("InsertRandomUniform")]
    | Annotated[Unary, Tag("Unary")],
    UnionDiscriminator(),
    UnionSerializer(),
]
"""Type alias for all supported transformation configurations."""

__all__ = ["Transform"]
