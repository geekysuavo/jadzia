"""Composite loss configuration."""

# Standard
from collections.abc import Callable
from typing import ClassVar, cast

# Third party
import equinox as eqx

# Dax
from jadzia.config.loss.function import LossFunction
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.mapping import MappingConfig
from jadzia.nn.loss.composite import CompositeLoss as CompositeLossImpl
from jadzia.nn.loss.composite import CompositeLossTerm


class CompositeLoss(MappingConfig[LossFunction]):
    """
    Configuration for [jadzia.nn.loss.composite.CompositeLoss][].

    Attributes:
        terms: Names of each term and their associated weights.
        functions: Definitions of each term in the weighted sum.
    """

    instance_type: ClassVar[type[eqx.Module]] = CompositeLossImpl

    @property
    def prediction_names(self) -> tuple[Identifier, ...]:
        """Names of predicted values passed as inputs to this loss."""
        all_names = (func.prediction_names for func in self.values())
        return tuple(sorted(set().union(*all_names)))

    @property
    def reference_names(self) -> tuple[Identifier, ...]:
        """Names of reference values passed as inputs to this loss."""
        all_names = (func.reference_names for func in self.values())
        return tuple(sorted(set().union(*all_names)))

    def build_instance(self) -> CompositeLossImpl:
        """Build an instance based on this configuration."""
        terms = tuple(
            CompositeLossTerm(
                weight=self[name].weight,
                prediction_names=tuple(map(str, self[name].prediction_names)),
                reference_names=tuple(map(str, self[name].reference_names)),
                module=cast(Callable, self[name].build_instance()),
            )
            for name in sorted(self)
        )
        return CompositeLossImpl(terms=terms)


__all__ = ["CompositeLoss"]
