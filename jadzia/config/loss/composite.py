"""Composite loss configuration."""

# Standard
from collections.abc import Callable
from typing import ClassVar, cast

# Third party
import equinox as eqx

# Dax
from jadzia.config.loss.function import LossFunction
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, Reserved
from jadzia.core.validation.context import WithRuntimeContext
from jadzia.core.validation.mapping import MappingConfig
from jadzia.nn.loss.composite import CompositeLoss as CompositeLossImpl
from jadzia.nn.loss.composite import CompositeLossTerm


class CompositeLoss(MappingConfig[LossFunction], WithRuntimeContext):
    """
    Configuration for [jadzia.nn.loss.composite.CompositeLoss][].

    Attributes:
        terms: Names of each term and their associated weights.
        functions: Definitions of each term in the weighted sum.
    """

    instance_type: ClassVar[type[eqx.Module]] = CompositeLossImpl

    def build_instance(self) -> CompositeLossImpl:
        """Build an instance based on this configuration."""
        terms = tuple(
            CompositeLossTerm(
                name=name,
                weight=self[name].weight,
                prediction_names=tuple(map(str, self[name].prediction_names)),
                reference_names=tuple(map(str, self[name].reference_names)),
                module=cast(Callable, self[name].build_instance()),
            )
            for name in sorted(self)
        )
        module = CompositeLossImpl(terms=terms)
        return eqx.filter_shard(module, self.context.model_sharding)

    def build_subgraph(self) -> tuple[Graph[Identifier], Identifier]:
        """
        Build a dependency graph based on this configuration.

        Returns:
            Graph containing the loss function subgraph.
            Terminal graph node containing the final loss function value.
        """
        outputs = []
        graph = Graph[Identifier]()
        for name, config in self.items():
            node = Identifier.gather(name)
            output = Identifier(name, "output")
            graph[node] = config.prediction_names + config.reference_names
            graph[output] = (node,)
            outputs.append(output)

        node = Identifier.gather(Reserved.LOSS)
        graph[node] = outputs

        terminal = Identifier(Reserved.LOSS, "output")
        graph[terminal] = (node,)
        return (graph, terminal)


__all__ = ["CompositeLoss"]
