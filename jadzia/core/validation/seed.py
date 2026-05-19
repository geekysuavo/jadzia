"""Random seed validation."""

# Standard
import os
import sys
from typing import Annotated

# Third party
from pydantic import Field, PositiveInt


def _default_random_seed() -> int:
    """Return a reasonably random number."""
    return int.from_bytes(os.urandom(4), byteorder=sys.byteorder, signed=False)


RandomSeed = Annotated[PositiveInt, Field(default_factory=_default_random_seed)]
"""Type annotation for random seeds that are realized during validation if unset."""

__all__ = ["RandomSeed"]
