"""Directed acyclic graphs for dependency management."""

# Standard
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

# Dax
from jadzia.core.dependency.identifier import Identifier


class Graph[NodeT: str | Identifier]:
    """A directed acyclic graph (DAG) of execution dependencies."""

    def __init__(self, dependencies: Mapping[NodeT, Iterable[NodeT]] | None = None):
        """
        Initialize the dependency graph.

        Args:
            dependencies: An optional mapping from graph nodes to the nodes
                that they depend on.
        """
        dependencies = dependencies or {}
        self._deps = {node: set(deps) for node, deps in dependencies.items()}

    def __str__(self) -> str:
        """String representation of the graph."""
        lines = ["Graph("]
        lines.extend(
            f"  {node} : ({', '.join(map(str, sorted(deps)))})"
            for node, deps in self._deps.items()
        )
        lines.append(")")
        return "\n".join(lines)

    def __eq__(self, other: Any) -> bool:
        """
        Whether two graphs are equivalent.

        Args:
            other: Graph to compare against.

        Returns:
            Whether this graph is equivalent to `other`.
        """
        if type(self) is type(other):
            return self.dependencies == other.dependencies
        else:
            return False

    def __len__(self) -> int:
        """
        Size of the graph.

        Returns:
            Number of graph nodes.
        """
        return len(self._deps)

    def __iter__(self) -> Iterator[NodeT]:
        """
        Iterate over graph nodes.

        Returns:
            Iterator over graph nodes.
        """
        return iter(self._deps)

    def __contains__(self, node: NodeT) -> bool:
        """
        Whether the graph contains a given node.

        Args:
            node: Node to check.

        Returns:
            True if the graph contains `node`.
        """
        return node in self._deps

    def __getitem__(self, node: NodeT) -> set[NodeT]:
        """
        Get the dependencies of a graph node.

        Args:
            node: Node to access.

        Returns:
            Dependencies of `node` in the graph.

        Raises:
            KeyError: If `node` is not contained in the graph.
        """
        return self._deps[node]

    def __setitem__(self, node: NodeT, deps: Iterable[NodeT]):
        """
        Set the dependencies of a graph node.

        Args:
            node: Node to modify.
            deps: New dependencies to assign.

        Note:
            If any nodes in `deps` are not present in the graph, they will be added.
        """
        depset = set(deps)
        self._deps[node] = depset
        extra_nodes = depset - set(self._deps)
        self._deps.update({other: set() for other in extra_nodes})

    def __delitem__(self, node: NodeT):
        """
        Remove a graph node.

        Args:
            node: Node the remove.

        Raises:
            KeyError: If `node` is not contained in the graph.

        Note:
            The removed `node` will also be removed from all dependency sets.
        """
        del self._deps[node]
        self._deps = {other: deps - {node} for other, deps in self._deps.items()}

    def update(self, node: NodeT, *deps: NodeT):
        """
        Add dependencies to a graph node.

        Args:
            node: Node to update.
            deps: Additional dependencies to assign.

        Note:
            If `node` or any node in `deps` are not present in the graph, they
            will be added.
        """
        self[node] = self._deps.get(node, set()) | set(deps)

    @property
    def dependencies(self) -> tuple[tuple[NodeT, tuple[NodeT, ...]], ...]:
        """Canonicalized immutable form of the dependency graph."""
        items = sorted(
            (node, tuple(sorted(deps)))
            for node, deps in self._deps.items()
            if len(deps) > 0
        )
        return tuple(items)

    def sort_nodes(self, subset: Iterable[NodeT] | None = None) -> tuple[NodeT, ...]:
        """
        Topologically sort graph nodes or a subset of graph nodes.

        Args:
            subset: Subset of required graph nodes used to initiate traversal.
                If unspecified, all nodes in the graph will be required.

        Returns:
            Tuple of nodes in topologically sorted order. Visiting nodes in this order
                will ensure that their dependencies are ready to use when they are
                visited.
        """
        unvisited = sorted(subset or self)
        visited: list[NodeT] = []
        marked: list[NodeT] = []
        order: list[NodeT] = []

        def visit(node: NodeT):
            if node in visited:
                return
            elif node in marked:
                raise ValueError(f"graph is not a DAG:\n{self}")
            elif node not in self:
                raise ValueError(f"graph does not contain node: {node}")

            marked.append(node)
            neighbors = sorted(self[node])
            for neighbor in neighbors:
                visit(neighbor)

            marked.remove(node)
            visited.append(node)
            order.append(node)

        while len(unvisited):
            node = unvisited.pop()
            visit(node)

        return tuple(order)


__all__ = ["Graph"]
