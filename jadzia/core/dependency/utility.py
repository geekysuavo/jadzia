"""Dependency management utility functions."""

# Standard
import functools
import itertools
import operator
from collections.abc import Iterable
from typing import Literal, overload

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier


def is_gather(identifier: Identifier) -> bool:
    """Return whether an identifier is of the 'gather' type."""
    return identifier.is_gather


def to_node(identifier: Identifier) -> str:
    """Return the node name of an identifier."""
    return identifier.node


def merge_graphs(*graphs: Graph[Identifier]) -> Graph[Identifier]:
    """
    Compute the union of one or more identifier graphs.

    Args:
        graphs: Graphs to merge together.

    Returns:
        Union of all graphs.

    Raises:
        ValueError: If any two graphs contain identical gather nodes.
    """
    overlaps = set()
    for first, second in itertools.combinations(graphs, 2):
        overlaps.update(filter(is_gather, first & second))

    if overlaps:
        overlap_str = ", ".join(map(str, overlaps))
        raise ValueError(f"names are not unique: {overlap_str}")

    return functools.reduce(operator.or_, graphs)


def sort_gathers(
    graph: Graph[Identifier],
    subset: Iterable[Identifier] | None = None,
) -> tuple[str, ...]:
    """
    Topologically sort the gather nodes in an identifier graph.

    Args:
        graph: Graph to topologically sort.
        subset: Subset of required identifiers.

    Returns:
        Gather nodes in topologically sorted order.
    """
    nodes = graph.sort_nodes(subset)
    return tuple(map(to_node, filter(is_gather, nodes)))


@overload
def find_unused(
    graph: Graph[Identifier],
    subset: Iterable[Identifier] | None = None,
    *,
    gathers: Literal[True] = True,
) -> set[str]: ...


@overload
def find_unused(
    graph: Graph[Identifier],
    subset: Iterable[Identifier] | None = None,
    *,
    gathers: Literal[False] = False,
) -> set[Identifier]: ...


def find_unused(
    graph: Graph[Identifier],
    subset: Iterable[Identifier] | None = None,
    *,
    gathers: bool = True,
) -> set[str] | set[Identifier]:
    """
    Find the unused nodes in an identifier graph.

    Args:
        graph: Graph to topologically sort.
        subset: Subset of required identifiers.
        gathers: When true, only return unused gather nodes.

    Returns:
        Unused nodes, determined by first sorting the graph
            and then subtracting the result from all graph nodes.
    """
    if gathers:
        all_nodes = set(map(to_node, filter(is_gather, graph)))
        used_nodes = set(sort_gathers(graph, subset))
        return all_nodes - used_nodes
    else:
        return set(graph) - set(graph.sort_nodes(subset))


__all__ = [
    "find_unused",
    "is_gather",
    "merge_graphs",
    "sort_gathers",
    "to_node",
]
