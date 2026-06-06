"""Tests of validated transformation classes."""

# Standard
import math
from typing import NamedTuple

# Third party
import pytest
from numpy.random import Generator
from pydantic import PositiveInt

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.transform import TransformConfig, TransformSignature
from jadzia.data.transform.base import Map, MapWithIndex, RandomMap


class Answer(Map):
    def __call__(self) -> int:
        return 42


class Multiply(Map):
    n: PositiveInt

    def __call__(self, x: int) -> int:
        return self.n * x


class Step(MapWithIndex):
    def __call__(self, z: int, *, index: int) -> int:
        return z + index % 4


class RandomScale(RandomMap):
    loc: float = 0.0
    scale: float = 1.0

    def __call__(self, p: float, *, rng: Generator) -> float:
        return p * rng.normal(self.loc, self.scale).item()


class Rotate(Map):
    class ReturnType(NamedTuple):
        arot: float
        brot: float

    angle: float = 1.0

    def __call__(self, a: float, b: float) -> ReturnType:
        (cos, sin) = (math.cos(self.angle), math.sin(self.angle))
        arot = cos * a - sin * b
        brot = cos * b + sin * a
        return self.ReturnType(arot, brot)


class InvalidIndex(MapWithIndex):
    def __call__(self, index: int) -> int: ...


class InvalidRng(RandomMap):
    def __call__(self, rng: Generator, /) -> int: ...


def test_map_no_input():
    """Inspection of map transformations produces the expected result."""
    config_cls = TransformConfig.auto(Answer)
    assert issubclass(config_cls, TransformConfig)
    assert config_cls.instance_call.requires_index is False
    assert config_cls.instance_call.requires_rng is False

    config = config_cls()
    assert config.input_names == ()
    assert config.output_names == ("output",)

    with pytest.raises(ValueError, match="accepts no inputs"):
        config_cls(input="a.b")

    with pytest.raises(ValueError, match="accepts no inputs"):
        config_cls(inputs={"x": "a.b"})


def test_map_single_output():
    """Inspection of map transformations produces the expected result."""
    config_cls = TransformConfig.auto(Multiply)
    assert issubclass(config_cls, TransformConfig)
    assert config_cls.instance_call.requires_index is False
    assert config_cls.instance_call.requires_rng is False

    config = config_cls(input="tform.result", n=2)
    assert config.input_names == (Identifier("tform", "result"),)
    assert config.output_names == ("output",)

    obj = config.build_instance(name="mul")
    assert isinstance(obj, Multiply)
    assert obj.input_names == ("tform.result",)
    assert obj.output_names == ("mul.output",)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()

    with pytest.raises(ValueError, match="requires exactly one"):
        config_cls(n=4)

    with pytest.raises(ValueError, match="Field required"):
        config_cls(input="a.b")

    with pytest.raises(ValueError, match="requires an `input`"):
        config_cls(inputs={}, n=1)


def test_map_with_index():
    """Inspection of indexed map transformations produces the expected result."""
    config_cls = TransformConfig.auto(Step)
    assert issubclass(config_cls, TransformConfig)
    assert config_cls.instance_call.requires_index is True
    assert config_cls.instance_call.requires_rng is False

    config = config_cls(input="tform.res")
    assert config.input_names == (Identifier("tform", "res"),)
    assert config.output_names == ("output",)

    obj = config.build_instance(name="step")
    assert isinstance(obj, Step)
    assert obj.input_names == ("tform.res",)
    assert obj.output_names == ("step.output",)


def test_map_multi_output():
    """Inspection of multi-output call methods produces the expected result."""
    config_cls = TransformConfig.auto(Rotate)
    assert issubclass(config_cls, TransformConfig)
    assert config_cls.instance_call.requires_index is False
    assert config_cls.instance_call.requires_rng is False

    config = config_cls(inputs={"a": "first.out", "b": "second.out"})
    assert config.input_names == (
        Identifier("first", "out"),
        Identifier("second", "out"),
    )
    assert config.output_names == ("arot", "brot")

    obj = config.build_instance(name="rot")
    assert isinstance(obj, Rotate)
    assert obj.input_names == ("first.out", "second.out")
    assert obj.output_names == ("rot.arot", "rot.brot")

    with pytest.raises(ValueError, match="requires `inputs`"):
        config_cls(input="a.b")

    with pytest.raises(ValueError, match="invalid transformation call inputs"):
        config_cls(inputs={"c": "third.out"})


def test_random_map():
    """Inspection of random map transformations produces the expected result."""
    config_cls = TransformConfig.auto(RandomScale)
    assert issubclass(config_cls, TransformConfig)
    assert config_cls.instance_call.requires_index is False
    assert config_cls.instance_call.requires_rng is True

    config = config_cls(input="tf.rv")
    assert config.input_names == (Identifier("tf", "rv"),)
    assert config.output_names == ("output",)

    obj = config.build_instance(name="scale")
    assert isinstance(obj, RandomScale)
    assert obj.input_names == ("tf.rv",)
    assert obj.output_names == ("scale.output",)


def test_transform_signature():
    """Transformation signatures satisfy a minimal set of requirements."""
    with pytest.raises(ValueError, match="index parameter"):
        TransformSignature.from_type(InvalidIndex)

    with pytest.raises(ValueError, match="rng parameter"):
        TransformSignature.from_type(InvalidRng)
