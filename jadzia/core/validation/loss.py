"""Validation base classes for loss functions."""

# Standard
from abc import ABC, abstractmethod
from typing import Annotated, Self

# Third party
from pydantic import model_validator

# Dax
from jadzia.core.validation.buildable import Exclude
from jadzia.core.validation.module import ModuleConfig


class LossConfig(ModuleConfig, ABC):
    """Loss function configuration base class."""

    @property
    @abstractmethod
    def prediction_names(self) -> tuple[str, ...]:
        """Names of predicted values passed as inputs to this loss."""

    @property
    @abstractmethod
    def reference_names(self) -> tuple[str, ...]:
        """Names of reference values passed as inputs to this loss."""

    @model_validator(mode="after")
    def _validate_output_signature(self) -> Self:
        """Ensure the loss function returns a single output value."""
        if len(self.instance_call.outputs) != 1:
            raise ValueError(
                f"expected one module call output in {type(self).__name__}, "
                f"got {len(self.instance_call.outputs)} outputs"
            )
        else:
            return self


class SimpleLoss(LossConfig):
    """
    Loss function taking a single prediction-reference pair.

    Modules built by this loss must have a single positional-only argument
    and a single positional-or-keyword argument, like so:
    ```python
    def __call__(self, yhat: Array, /, y: Array) -> Array:
    ```

    Attributes:
        input: Name of the input to the loss function.
    """

    input: Annotated[str, Exclude]

    @property
    def prediction_names(self) -> tuple[str, ...]:
        """Names of predicted values passed as inputs to this loss."""
        return (self.input,)

    @property
    def reference_names(self) -> tuple[str, ...]:
        """Names of reference values passed as inputs to this loss."""
        return (self.input,)

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required number and kind of inputs is provided."""
        if (
            len(self.instance_call.inputs_pos_only) != 1
            or len(self.instance_call.inputs_pos_or_kw) != 1
        ):
            raise ValueError(
                f"expected one module call input pair in {type(self).__name__}, "
                f"got {len(self.instance_call.inputs_pos_only)} positional only "
                f"and {len(self.instance_call.inputs_pos_or_kw)} positional or keyword"
            )
        else:
            return self


class GeneralLoss(LossConfig):
    """
    Loss function taking multiple prediction and/or reference inputs.

    Modules built by this loss may have one or more positional-only arguments
    and must have one or more positional-or-keyword arguments, like so:
    ```python
    def __call__(self, *yhats: Array) -> Array:
    def __call__(self, *yhats: Array, /, *ys: Array) -> Array:
    ```

    Attributes:
        inputs: Mapping from module call method argument names to input names.
    """

    inputs: Annotated[dict[str, str], Exclude]

    @property
    def prediction_names(self) -> tuple[str, ...]:
        """Names of predicted values passed as inputs to this loss."""
        if len(self.instance_call.inputs_pos_only):
            original_names = self.instance_call.inputs_pos_only
        else:
            original_names = self.instance_call.inputs_pos_or_kw

        return tuple(self.inputs[name] for name in original_names)

    @property
    def reference_names(self) -> tuple[str, ...]:
        """Names of reference values passed as inputs to this loss."""
        if len(self.instance_call.inputs_pos_only):
            original_names = self.instance_call.inputs_pos_or_kw
        else:
            original_names = ()

        return tuple(self.inputs[name] for name in original_names)

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required number and kind of inputs is provided."""
        supplied_names = set(self.inputs)
        required_names = set().union(
            self.instance_call.inputs_pos_only,
            self.instance_call.inputs_pos_or_kw,
        )
        missing_names = required_names - supplied_names
        extra_names = supplied_names - required_names
        if missing_names or extra_names:
            raise ValueError(
                f"invalid module call inputs in {type(self).__name__}; "
                f"missing: ({', '.join(missing_names)}), "
                f"extra: ({', '.join(extra_names)})"
            )
        else:
            return self


__all__ = ["GeneralLoss", "LossConfig", "SimpleLoss"]
