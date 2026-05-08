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


class BuildableWidget(BuildableConfig[Widget]): ...


class FooConfig(BuildableWidget):
    instance_type: ClassVar[type[Widget]] = FooWidget
    extra: Annotated[str, Exclude] = "hello"
    n: int = 3


BarConfig = BuildableWidget.auto(BarWidget)


class ParentConfig(BuildableConfig[dict]):
    foo: FooConfig = FooConfig()
    bar: BarConfig = BarConfig()


@pytest.mark.parametrize("n", [3, 5, 7])
def test_foo(n):
    """Widgets may be built from manually written configuration classes."""
    config = FooConfig(n=n)
    widget = config.build_instance()
    assert isinstance(widget, FooWidget)
    assert isinstance(widget.do_something("example"), str)


@pytest.mark.parametrize("flag", [True, False])
def test_bar(flag):
    """Widgets may be built from automatically created configuration classes."""
    config = BarConfig(flag=flag)
    widget = config.build_instance()
    assert isinstance(widget, BarWidget)
    assert isinstance(widget.do_something("example"), str)


def test_model_dump_for_instance():
    """Dumping data for building instances produces the expected results."""
    foo = FooConfig()
    bar = BarConfig()
    config = ParentConfig()

    data = foo.model_dump_for_instance()
    assert set(data) == {"n"}  # extra is excluded

    data = bar.model_dump_for_instance()
    assert set(data) == {"flag"}  # auto classes work too

    data = config.model_dump_for_instance(pass_subclasses=(FooConfig,))
    assert isinstance(data["foo"], FooConfig)  # passed
    assert isinstance(data["bar"], dict)  # dumped

    data = config.model_dump_for_instance(build_subclasses=(BarConfig,))
    assert isinstance(data["foo"], dict)  # dumped
    assert isinstance(data["bar"], BarWidget)  # built
