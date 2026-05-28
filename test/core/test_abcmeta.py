"""Tests of the abstract dataclass metaclass."""

# Standard
from abc import abstractmethod
from dataclasses import FrozenInstanceError

# Third party
import pytest

# Dax
from jadzia.core.meta import AbstractDataclassMeta


class Abstract(metaclass=AbstractDataclassMeta):
    x: int = 2

    @abstractmethod
    def func(self): ...


class Concrete(Abstract):
    y: float = 3.25

    def func(self) -> float:
        return self.x * self.y


def test_abstract():
    """Instantiating an abstract class raises an exception."""
    with pytest.raises(TypeError, match="instantiate abstract class"):
        Abstract()


def test_dataclass():
    """Classes using the metaclass obey composition."""
    obj = Concrete()
    assert obj.x == 2
    assert obj.y == 3.25
    assert obj.func() == 6.5


def test_frozen():
    """Classes using the metaclass are frozen after creation."""
    obj = Concrete()
    with pytest.raises(FrozenInstanceError):
        obj.x = 3
