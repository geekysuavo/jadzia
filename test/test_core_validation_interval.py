"""Tests of interval types."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.core.validation.interval import (
    UnitInterval,
    UnitIntervalOpen,
    UnitIntervalOpenLeft,
    UnitIntervalOpenRight,
)


def test_closed():
    """Closed intervals allow both endpoints."""
    adapter = TypeAdapter(UnitInterval)

    assert adapter.validate_python(0) == 0.0
    assert adapter.validate_python(1) == 1.0
    assert adapter.validate_python(0.5) == 0.5

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        adapter.validate_python(-0.1)

    with pytest.raises(ValueError, match="less than or equal to 1"):
        adapter.validate_python(1.1)


def test_open():
    """Open intervals allow neither endpoint."""
    adapter = TypeAdapter(UnitIntervalOpen)

    assert adapter.validate_python(0.5) == 0.5

    with pytest.raises(ValueError, match="greater than 0"):
        adapter.validate_python(0)

    with pytest.raises(ValueError, match="less than 1"):
        adapter.validate_python(1)


def test_open_left():
    """Open-left intervals allow the right endpoint."""
    adapter = TypeAdapter(UnitIntervalOpenLeft)

    assert adapter.validate_python(1) == 1.0
    assert adapter.validate_python(0.5) == 0.5

    with pytest.raises(ValueError, match="greater than 0"):
        adapter.validate_python(0)

    with pytest.raises(ValueError, match="less than or equal to 1"):
        adapter.validate_python(1.1)


def test_open_right():
    """Open-right intervals allow the left endpoint."""
    adapter = TypeAdapter(UnitIntervalOpenRight)

    assert adapter.validate_python(0) == 0.0
    assert adapter.validate_python(0.5) == 0.5

    with pytest.raises(ValueError, match="greater than or equal to 0"):
        adapter.validate_python(-0.1)

    with pytest.raises(ValueError, match="less than 1"):
        adapter.validate_python(1)
