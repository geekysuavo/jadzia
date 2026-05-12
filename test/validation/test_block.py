"""Tests of validated model building block classes."""

# Standard
from typing import NamedTuple

# Third party
import equinox as eqx
import jax
import pytest

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.block import (
    FissionBlock,
    FusionBlock,
    GeneralBlock,
    SimpleBlock,
)


class General(eqx.Module):
    class ReturnType(NamedTuple):
        foo: jax.Array
        bar: jax.Array

    def __call__(self, x: jax.Array, y: jax.Array) -> ReturnType:
        return self.ReturnType(x + y, x - y)


class Simple(eqx.Module):
    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.numpy.square(x)


class Fusion(eqx.Module):
    def __call__(self, *args: jax.Array) -> jax.Array:
        return jax.numpy.stack(args, axis=0).mean(axis=0)


class Fission(eqx.Module):
    def __call__(self, x: jax.Array) -> tuple[jax.Array, ...]:
        return jax.numpy.unstack(x)


class PositionalOnly(eqx.Module):
    def __call__(self, arg: jax.Array, /) -> jax.Array:
        return jax.numpy.sum(arg)


def test_general_block():
    """Signature discovery of general blocks produces the expected results."""
    config_cls = GeneralBlock.auto(General)
    assert issubclass(config_cls, GeneralBlock)

    config = config_cls(inputs={"x": "batch.hello", "y": "batch.world"})
    assert config.input_names == (
        Identifier("batch", "hello"),
        Identifier("batch", "world"),
    )
    assert config.output_names == ("foo", "bar")

    obj = config.build_instance()
    assert isinstance(obj, General)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()

    with pytest.raises(ValueError, match="missing module call inputs"):
        config_cls(inputs={})

    with pytest.raises(ValueError, match="extra module call inputs"):
        config_cls(inputs={"x": "a.b", "y": "c.d", "extra": "e.f"})


def test_simple_block():
    """Signature discovery of simple blocks produces the expected results."""
    config_cls = SimpleBlock.auto(Simple)
    assert issubclass(config_cls, SimpleBlock)

    config = config_cls(input="blk.pred")
    assert config.input_names == (Identifier("blk", "pred"),)
    assert config.output_names == ("output",)

    obj = config.build_instance()
    assert isinstance(obj, Simple)

    with pytest.raises(ValueError, match="Field required"):
        config_cls()


def test_fusion_block():
    """Signature discovery of fusion blocks produces the expected results."""
    config_cls = FusionBlock.auto(Fusion)
    assert issubclass(config_cls, FusionBlock)

    config = config_cls(inputs=("mlp.out", "agg.mean", "agg.var"))
    assert config.input_names == (
        Identifier("mlp", "out"),
        Identifier("agg", "mean"),
        Identifier("agg", "var"),
    )
    assert config.output_names == ("output",)


def test_fission_block():
    """Signature discovery of fission blocks produces the expected results."""
    config_cls = FissionBlock.auto(Fission)
    assert issubclass(config_cls, FissionBlock)

    config = config_cls(input="mlp.out", outputs=("foo", "bar"))
    assert config.input_names == (Identifier("mlp", "out"),)
    assert config.output_names == ("foo", "bar")

    config = config_cls(input="mlp.out", num_outputs=3)
    assert config.input_names == (Identifier("mlp", "out"),)
    assert config.output_names == ("output1", "output2", "output3")


def test_fission_output_names():
    """Fission blocks validate mutual exclusivity of their output fields."""
    with pytest.raises(ValueError, match="requires exactly one of"):
        FissionBlock.auto(Fission)(input="foo.bar")

    with pytest.raises(ValueError, match="requires exactly one of"):
        FissionBlock.auto(Fission)(input="foo.bar", outputs=())

    with pytest.raises(ValueError, match="requires exactly one of"):
        FissionBlock.auto(Fission)(input="foo.bar", outputs=("hello",), num_outputs=1)

    # we have to bypass validation to reach the invalid state being tested.
    config = FissionBlock.auto(Fission).model_construct({"input": "a.b"})
    with pytest.raises(RuntimeError, match="contains invalid data"):
        config.output_names


def test_incompatible_signatures_fail():
    """Signatures are checked for compatibility with block config classes."""
    with pytest.raises(ValueError, match="expected one module call input"):
        SimpleBlock.auto(General)(input="a.b")

    with pytest.raises(ValueError, match="expected one module call input"):
        SimpleBlock.auto(Fusion)(input="a.b")

    with pytest.raises(ValueError, match="does not support module call signature"):
        SimpleBlock.auto(PositionalOnly)(input="a.b")

    with pytest.raises(ValueError, match="does not support module call signature"):
        GeneralBlock.auto(PositionalOnly)(inputs={"arg": "a.b"})

    with pytest.raises(ValueError, match="does not support module call signature"):
        FusionBlock.auto(Simple)(inputs=["a.b"])

    with pytest.raises(ValueError, match="does not support module call signature"):
        FusionBlock.auto(General)(inputs=["a.b"])
