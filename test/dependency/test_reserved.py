"""Tests of identifier reserved node names."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.reserved import Reserved


@pytest.mark.parametrize("output", ["foo", "bar"])
@pytest.mark.parametrize("reserved", Reserved)
def test_reserved(reserved, output):
    """Identifiers may be built from reserved node names."""
    example = reserved(output)
    assert example == Identifier.from_str(f"{reserved}.{output}")
