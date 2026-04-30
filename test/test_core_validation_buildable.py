"""Tests of validated buildable classes."""

# Standard
from abc import ABC, abstractmethod
from typing import Annotated, ClassVar

# Third party
import pytest

# Dax
from jadzia.core.validation.buildable import BuildableConfig, Exclude


class Widget(ABC):
    @abstractmethod
    def do_something(self, value: str) -> str: ...


class FooWidget(Widget):
    def __init__(self, n: int = 3):
        self.n = n

    def do_something(self, value: str) -> str:
        return value * self.n


class BarWidget(Widget):
    def __init__(self, flag: bool = True):
        self.flag = flag

    def do_something(self, value: str) -> str:
        return "".join(reversed(value)) if self.flag else value


class BuildableWidget(BuildableConfig[Widget]):
    instance_type: ClassVar[type[Widget]]


class FooConfig(BuildableWidget):
    instance_type: ClassVar[type[Widget]] = FooWidget
    extra: Annotated[str, Exclude] = "hello"
    n: int = 3


BarConfig = BuildableWidget.auto(BarWidget)


@pytest.mark.parametrize("config_cls", [FooConfig, BarConfig], ids=["foo", "bar"])
def test_building(config_cls):
    """Widgets may be built from their configuration classes."""
    config = config_cls()
    widget = config.build_instance()
    assert isinstance(widget, config_cls.instance_type)
    assert isinstance(widget.do_something("example"), str)
