"""Functions for inspecting class initializer and call methods."""

# Standard
import inspect
from collections.abc import Callable
from typing import Any, NamedTuple


class CallableSignature(NamedTuple):
    """
    Summary of inputs and outputs produced by inspecting a callable.

    Attributes:
        inputs_with_defaults: Names of input parameters that have default values.
        inputs_pos_only: Names of input parameters that must be passed positionally.
        inputs_pos_or_kw: Names of input parameters that may be passed either
            positionally or by keyword.
        inputs_kw_only: Names of input parameters that must be passed by keyword.
        outputs: Names of output values, or `output` when the callable returns
            a single output.
    """

    inputs_with_defaults: tuple[str, ...]
    inputs_pos_only: tuple[str, ...]
    inputs_pos_or_kw: tuple[str, ...]
    inputs_kw_only: tuple[str, ...]
    outputs: tuple[str, ...]


def inspect_initializer(
    input_cls: type,
    *,
    field_defs: dict[str, tuple[Any, Any]],
    **defaults: Any,
):
    """
    Populate a dictionary with field definitions from an `__init__()` method.

    Args:
        input_cls: Class to inspect.
        field_defs: Dictionary to update with new field definitions.
            See `pydantic.create_model` for more details on structure.
        defaults: Optional default values of fields.

    Raises:
        ValueError: If default values are provided to parameters not in the signature.
    """
    defaults = defaults or {}

    signature = inspect.signature(input_cls)
    for name, param in signature.parameters.items():
        if name.startswith("_"):
            continue

        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        param_type = param.annotation
        if param_type is inspect._empty:
            continue

        param_default = defaults.get(name) or param.default
        if param_default is inspect._empty:
            param_default = Ellipsis

        field_defs[name] = (param_type, param_default)

    extra_defaults = set(defaults) - set(field_defs)
    if extra_defaults:
        raise ValueError(f"extra defaults provided: {', '.join(extra_defaults)}")


def inspect_callable(fn: Callable, *, method: bool = True) -> CallableSignature:
    """
    Inspect a function or method and return a summary of its inputs and outputs.

    Args:
        fn: Function or method to inspect.
        method: When true, `fn` is assumed to be a method.

    Returns:
        Summary of inputs and outputs of `fn`.
    """
    signature = inspect.signature(fn)
    named_parameters = tuple(signature.parameters.items())
    return_type = signature.return_annotation

    if method:
        # we are inspecting an unbound method; drop the 'self' parameter
        named_parameters = named_parameters[1:]

    inputs_with_defaults = tuple(
        name
        for name, param in named_parameters
        if param.default is not inspect.Parameter.empty
    )

    inputs_pos_only = tuple(
        name
        for name, param in named_parameters
        if param.kind is inspect.Parameter.POSITIONAL_ONLY
    )

    inputs_pos_or_kw = tuple(
        name
        for name, param in named_parameters
        if param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )

    inputs_kw_only = tuple(
        name
        for name, param in named_parameters
        if param.kind is inspect.Parameter.KEYWORD_ONLY
    )

    outputs: tuple[str, ...]
    if return_type is inspect._empty or return_type is None:
        outputs = ()  # no output
    elif isinstance(return_type, str):
        raise RuntimeError("string type annotations do not support inspection")
    elif hasattr(return_type, "__annotations__") and issubclass(return_type, tuple):
        outputs = tuple(return_type.__annotations__)  # named tuple of outputs
    elif hasattr(return_type, "__annotations__") and issubclass(return_type, dict):
        outputs = tuple(return_type.__annotations__)  # typed dictionary of outputs
    else:
        outputs = ("output",)  # single output

    return CallableSignature(
        inputs_with_defaults,
        inputs_pos_only,
        inputs_pos_or_kw,
        inputs_kw_only,
        outputs,
    )


__all__ = ["CallableSignature", "inspect_callable", "inspect_initializer"]
