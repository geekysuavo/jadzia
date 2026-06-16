"""Validated mapping classes."""

# Standard
from typing import Iterable, Iterator, cast

# Third party
from pydantic import ConfigDict, RootModel

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, Node, UnreservedStr
from jadzia.core.validation.base import Config


class MappingConfig[ValueT: Config](RootModel[dict[UnreservedStr, ValueT]]):
    """Mapping that holds configuration classes as values."""

    model_config = ConfigDict(frozen=True)

    root: dict[UnreservedStr, ValueT]

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

    def build_subgraph(self) -> Graph[Identifier]:
        """Build a dependency graph based on this configuration."""
        graph = Graph[Identifier]()
        for name, config in self.items():
            node = Identifier.gather(name)
            graph[node] = cast(Node, config).input_names
            for output in Identifier.from_outputs(name, cast(Node, config)):
                graph[output] = (node,)

        return graph


__all__ = ["MappingConfig", "NodeMappingConfig"]
