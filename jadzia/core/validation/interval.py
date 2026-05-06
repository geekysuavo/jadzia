"""Type definitions of unit intervals."""

# Standard
from typing import Annotated

# Third party
from pydantic import Field

type UnitInterval = Annotated[float, Field(ge=0, le=1)]
"""The closed unit interval, $[0,1]$."""

type UnitIntervalOpen = Annotated[float, Field(gt=0, lt=1)]
"""The open unit interval, $(0,1)$."""

type UnitIntervalOpenLeft = Annotated[float, Field(gt=0, le=1)]
"""The half-open unit interval, $(0,1]$."""

type UnitIntervalOpenRight = Annotated[float, Field(ge=0, lt=1)]
"""The half-open unit interval, $[0,1)$."""

__all__ = [
    "UnitInterval",
    "UnitIntervalOpen",
    "UnitIntervalOpenLeft",
    "UnitIntervalOpenRight",
]
