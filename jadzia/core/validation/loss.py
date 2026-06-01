"""Validation base classes for loss functions."""

# Standard
from typing import Annotated, Self

# Third party
from pydantic import NonNegativeFloat, model_validator

# Dax
from jadzia.core.dependency.identifier import Identifier, IdentifierStr
from jadzia.core.validation.buildable import Exclude
from jadzia.core.validation.module import ModuleConfig


class LossConfig(ModuleConfig):
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
        weight: Weighting factor used when combining loss terms together.
    """

    inputs: Annotated[dict[str, IdentifierStr], Exclude]
    weight: Annotated[NonNegativeFloat, Exclude] = 1.0

    @property
    def prediction_names(self) -> tuple[Identifier, ...]:
        """Names of predicted values passed as inputs to this loss."""
        if len(self.instance_call.inputs_pos_only):
            original_names = self.instance_call.inputs_pos_only
        else:
            original_names = self.instance_call.inputs_pos_or_kw

        return tuple(self.inputs[name] for name in original_names)

    @property
    def reference_names(self) -> tuple[Identifier, ...]:
        """Names of reference values passed as inputs to this loss."""
        if len(self.instance_call.inputs_pos_only):
            original_names = self.instance_call.inputs_pos_or_kw
        else:
            original_names = ()

        return tuple(self.inputs[name] for name in original_names)

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


__all__ = ["LossConfig"]
