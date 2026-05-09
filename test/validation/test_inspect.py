"""Tests of signature inspection functions."""

# Standard
from typing import Any, NamedTuple, TypedDict

# Third party
import pytest

# Dax
from jadzia.core.validation.inspect import inspect_callable, inspect_initializer


class ExampleCls:
    class TupleOutputs(NamedTuple):
        foo: int
        bar: int

    class DictOutputs(TypedDict):
        oof: int
        rab: int

    def __init__(
        self,
        unannotated,
        _private: str,
        without_default: int,
        with_default: float = 2.5,
        *args: Any,
        **kwargs: Any,
    ):
        pass

    def method_no_return(self):
        pass

    def method_none_return(self) -> None:
        pass

    def method_str_annotation(self) -> "str":
        return "foo"

    def method_one_output(self, a: int, /, b: float, *, c: str) -> str:
        return ""

    def method_tuple_outputs(self, p: int, q: int) -> TupleOutputs:
        return self.TupleOutputs(3, 5)

    def method_dict_outputs(self, pp: int, qq: int) -> DictOutputs:
        return self.DictOutputs(oof=3, rab=5)


def example_function(fee: int, /, fie: float, *, foe: str) -> set:
    return {fee, fie, foe}


def test_inspect_initializer():
    """Class initializers may be inspected."""
    field_defs = {}
    inspect_initializer(ExampleCls, field_defs=field_defs)

    assert field_defs["without_default"] == (int, ...)
    assert field_defs["with_default"] == (float, 2.5)
    assert "unannnotated" not in field_defs
    assert "_private" not in field_defs
    assert "args" not in field_defs
    assert "kwargs" not in field_defs

    with pytest.raises(ValueError, match="extra defaults provided"):
        inspect_initializer(ExampleCls, field_defs={}, not_a_param=42)


def test_inspect_method():
    """Methods may be inspected without class instantiation."""
    # without return types, or explicitly returning None.
    signature = inspect_callable(ExampleCls.method_none_return)
    assert inspect_callable(ExampleCls.method_no_return) == signature
    assert signature.inputs_with_defaults == ()
    assert signature.inputs_pos_only == ()
    assert signature.inputs_pos_or_kw == ()
    assert signature.inputs_kw_only == ()
    assert signature.outputs == ()

    # returning a single output.
    signature = inspect_callable(ExampleCls.method_one_output)
    assert signature.inputs_with_defaults == ()
    assert signature.inputs_pos_only == ("a",)
    assert signature.inputs_pos_or_kw == ("b",)
    assert signature.inputs_kw_only == ("c",)
    assert signature.outputs == ("output",)

    # returning a named tuple of outputs.
    signature = inspect_callable(ExampleCls.method_tuple_outputs)
    assert signature.inputs_with_defaults == ()
    assert signature.inputs_pos_only == ()
    assert signature.inputs_pos_or_kw == ("p", "q")
    assert signature.inputs_kw_only == ()
    assert signature.outputs == ("foo", "bar")

    # returning a typed dictionary of outputs.
    signature = inspect_callable(ExampleCls.method_dict_outputs)
    assert signature.inputs_with_defaults == ()
    assert signature.inputs_pos_only == ()
    assert signature.inputs_pos_or_kw == ("pp", "qq")
    assert signature.inputs_kw_only == ()
    assert signature.outputs == ("oof", "rab")

    # future annotations are not yet supported.
    with pytest.raises(RuntimeError, match="string type annotations"):
        inspect_callable(ExampleCls.method_str_annotation)


def test_inspect_function():
    """Free functions may also be inspected."""
    signature = inspect_callable(example_function, method=False)
    assert signature.inputs_with_defaults == ()
    assert signature.inputs_pos_only == ("fee",)
    assert signature.inputs_pos_or_kw == ("fie",)
    assert signature.inputs_kw_only == ("foe",)
    assert signature.outputs == ("output",)
