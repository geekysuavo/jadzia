"""Validated mapping classes."""

# Standard
from typing import Iterable, Iterator, NamedTuple, cast

# Third party
from pydantic import ConfigDict, RootModel

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, Node
from jadzia.core.validation.base import Config


class MappingConfig[ValueT: Config](RootModel[dict[str, ValueT]]):
    """Mapping that holds configuration classes as values."""

    model_config = ConfigDict(frozen=True)

    root: dict[str, ValueT]

    def __len__(self) -> int:
        """
        Size of the mapping.

        Returns:
            Number of mapping entries.
        """
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # ty: ignore[invalid-method-override]
        """
        Iterate over mapping keys.

        Returns:
            Iterator over mapping keys.
        """
        return iter(self.root)

    def __contains__(self, key: str) -> bool:
        """
        Whether the mapping contains a given key.

        Args:
            key: Key to check.

        Returns:
            True if the mapping contains `key`.
        """
        return key in self.root

    def __getitem__(self, key: str) -> ValueT:
        """
        Get a mapping configuration entry.

        Args:
            key: Key to access.

        Returns:
            Configuration entry associated with `key`.

        Raises:
            KeyError: If `key` is not contained in the mapping.
        """
        return self.root[key]

    def keys(self) -> Iterable[str]:
        """Iterate over mapping keys."""
        return self.root.keys()

    def values(self) -> Iterable[ValueT]:
        """Iterate over mapping values."""
        return self.root.values()

    def items(self) -> Iterable[tuple[str, ValueT]]:
        """Iterate over mapping key-value pairs."""
        return self.root.items()


class NodeMappingConfig[ValueT: Config](MappingConfig[ValueT]):
    """Mapping that holds a directed acyclic graph of configuration classes."""

    class NodeNames(NamedTuple):
        """
        Summary of used, unused, and missing nodes and identifiers.

        Attributes:
            used: Used node names.
            unused: Unused node names.
            missing: Missing node outputs.
        """

        used: set[str]
        unused: set[str]
        missing: set[Identifier]

    @property
    def output_names(self) -> tuple[Identifier, ...]:
        """All outputs that are returned by the nodes."""
        node_outputs = {
            identifier
            for name, node in self.items()
            for identifier in Identifier.from_outputs(name, cast(Node, node))
        }
        return tuple(sorted(node_outputs))

    def node_names(self, outputs: Iterable[Identifier] | None = None) -> NodeNames:
        """
        Determine the used, unused and missing nodes in the graph.

        Args:
            outputs: Optional required outputs. Defaults to all outputs.

        Returns:
            Information about which nodes are used, unused or missing.
        """
        outputs_to_check = set(outputs or self.output_names)
        missing_outputs = set()
        visited_nodes = set()

        while outputs_to_check:
            output = outputs_to_check.pop()
            name = output.node

            if name in visited_nodes:
                continue

            if (
                name not in self
                or output.output not in cast(Node, self[name]).output_names
            ):
                missing_outputs.add(output)
                continue

            visited_nodes.add(name)
            outputs_to_check.update(cast(Node, self[name]).input_names)

        unvisited_nodes = set(self) - visited_nodes
        return self.NodeNames(visited_nodes, unvisited_nodes, missing_outputs)

    def node_order(self, subset: Iterable[str] | None = None) -> tuple[str, ...]:
        """
        Determine the node execution order.

        Args:
            subset: Optional subset of required node names. Defaults to all nodes.

        Returns:
            Sequence of node names in their execution order.
        """
        deps = {
            name: {dep.node for dep in cast(Node, config).input_names}
            for name, config in self.items()
        }
        return Graph(deps).sort_nodes(subset)


__all__ = ["MappingConfig", "NodeMappingConfig"]
