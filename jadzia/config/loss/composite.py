"""Composite loss configuration."""

# Standard
from collections.abc import Callable
from typing import Annotated, ClassVar, Self, cast

# Third party
import equinox as eqx
from pydantic import Field, NonNegativeFloat, model_validator

# Dax
from jadzia.config.loss.function import LossFunction
from jadzia.core.validation.loss import LossConfig
from jadzia.core.validation.module import ModuleSignature
from jadzia.nn.loss.composite import CompositeLoss as CompositeLossImpl
from jadzia.nn.loss.composite import CompositeLossTerm


class CompositeLoss(LossConfig):
    """
    Configuration for [jadzia.nn.loss.composite.CompositeLoss][].

    Attributes:
        terms: Names of each term and their associated weights.
        functions: Definitions of each term in the weighted sum.
    """

    instance_type: ClassVar[type[eqx.Module]] = CompositeLossImpl
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        CompositeLossImpl
    )

    terms: Annotated[dict[str, NonNegativeFloat], Field(min_length=1)]
    functions: dict[str, LossFunction]

    @property
    def prediction_names(self) -> tuple[str, ...]:
        """Names of predicted values passed as inputs to this loss."""
        all_names = (func.prediction_names for func in self.functions.values())
        return tuple(sorted(set().union(*all_names)))

    @property
    def reference_names(self) -> tuple[str, ...]:
        """Names of reference values passed as inputs to this loss."""
        all_names = (func.reference_names for func in self.functions.values())
        return tuple(sorted(set().union(*all_names)))

    def build_instance(self) -> eqx.Module:  # ty: ignore[invalid-method-override]
        """Build an instance based on this configuration."""
        terms = tuple(
            CompositeLossTerm(
                weight=weight,
                prediction_names=self.functions[name].prediction_names,
                reference_names=self.functions[name].reference_names,
                module=cast(Callable, self.functions[name].build_instance()),
            )
            for name, weight in self.terms.items()
        )
        return CompositeLossImpl(terms)

    @model_validator(mode="after")
    def _validate_functions_and_terms(self) -> Self:
        """Ensure that all term and function names match."""
        missing_names = set(self.terms) - set(self.functions)
        extra_names = set(self.functions) - set(self.terms)
        if missing_names or extra_names:
            raise ValueError(
                "invalid composite loss configuration; "
                f"undefined functions: ({', '.join(missing_names)}), "
                f"unused functions: ({', '.join(extra_names)})"
            )
        else:
            return self


__all__ = ["CompositeLoss"]
