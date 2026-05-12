"""Validation base classes for model building blocks."""

# Standard
from abc import ABC, abstractmethod
from typing import Annotated, Self

# Third party
from pydantic import PositiveInt, model_validator

# Dax
from jadzia.core.dependency.identifier import Identifier, IdentifierStr
from jadzia.core.validation.buildable import Exclude
from jadzia.core.validation.module import ModuleConfig


class BlockConfig(ModuleConfig, ABC):
    """Model building block configuration base class."""

    @property
    @abstractmethod
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of other block outputs that are passed as inputs to this block."""

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of outputs that are returned by this block."""
        return self.instance_call.outputs


class SimpleBlock(BlockConfig):
    """
    Model building block taking a single input.

    Attributes:
        input: Name of the input to the block in `block.output` format.
    """

    input: Annotated[IdentifierStr, Exclude]

    @property
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of other block outputs that are passed as inputs to this block."""
        return (self.input,)

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required number and kind of inputs is provided."""
        if len(self.instance_call.inputs_pos_only) > 0:
            raise ValueError(
                f"simple block class {type(self).__name__} does not support "
                f"module call signature ({self.instance_call})"
            )
        elif len(self.instance_call.inputs_pos_or_kw) != 1:
            raise ValueError(
                f"expected one module call input in {type(self).__name__}, "
                f"got {len(self.instance_call.inputs_pos_or_kw)}"
            )
        else:
            return self


class GeneralBlock(BlockConfig):
    """
    Model building block taking multiple inputs.

    Attributes:
        inputs: Mapping from module call method argument names to input names
            in `block.output` format.
    """

    inputs: Annotated[dict[str, IdentifierStr], Exclude]

    @property
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of other block outputs that are passed as inputs to this block."""
        original_names = self.instance_call.inputs_pos_or_kw
        return tuple(self.inputs[name] for name in original_names)

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required number and kind of inputs is provided."""
        if len(self.instance_call.inputs_pos_only) > 0:
            raise ValueError(
                f"general block class {type(self).__name__} does not support "
                f"module call signature ({self.instance_call})"
            )

        supplied_names = set(self.inputs)
        required_names = set(self.instance_call.inputs_pos_or_kw)

        missing_names = required_names - supplied_names
        if missing_names:
            raise ValueError(
                f"missing module call inputs in {type(self).__name__}: "
                f"({', '.join(missing_names)})"
            )

        extra_names = supplied_names - required_names
        if extra_names:
            raise ValueError(
                f"extra module call inputs in {type(self).__name__}: "
                f"({', '.join(extra_names)})"
            )

        return self


class FusionBlock(BlockConfig):
    """
    Model building block taking a variable number of inputs.

    Attributes:
        inputs: List of input names in `block.output` format.
    """

    inputs: Annotated[list[IdentifierStr], Exclude]

    @property
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of other block outputs that are passed as inputs to this block."""
        return tuple(self.inputs)

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required kind of inputs is provided."""
        if (
            len(self.instance_call.inputs_pos_only) > 0
            or len(self.instance_call.inputs_pos_or_kw) > 0
            or len(self.instance_call.inputs_kw_only) > 0
        ):
            raise ValueError(
                f"fusion block class {type(self).__name__} does not support "
                f"module call signature ({self.instance_call})"
            )
        else:
            return self


class FissionBlock(SimpleBlock):
    """
    Model building block returning a variable number of outputs.

    Attributes:
        input: Name of the input to the block in `block.output` format.
        outputs: Names of outputs that are returned by this block.
            Mutually exclusive with `num_outputs`.
        num_outputs: Number of outputs that are returned by this block.
            When specified, outputs are named `output1` through `outputN`.
            Mutually exclusive with `outputs.
    """

    outputs: Annotated[tuple[str, ...] | None, Exclude] = None
    num_outputs: Annotated[PositiveInt | None, Exclude] = None

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of outputs that are returned by this block."""
        if self.outputs is not None:
            return self.outputs
        elif self.num_outputs is not None:
            return tuple(f"output{index + 1}" for index in range(self.num_outputs))
        else:
            raise RuntimeError(
                f"fission block {type(self).__name__} contains invalid data"
            )

    @model_validator(mode="after")
    def _validate_output_names(self) -> Self:
        """Ensure only one output field is provided."""
        fields = {bool(self.outputs), self.num_outputs is not None}
        if len(fields) == 1:
            raise ValueError(
                f"fission block class {type(self).__name__} requires "
                f"exactly one of `outputs` and `num_outputs` to be specified"
            )
        else:
            return self


__all__ = ["BlockConfig", "FissionBlock", "FusionBlock", "GeneralBlock", "SimpleBlock"]
