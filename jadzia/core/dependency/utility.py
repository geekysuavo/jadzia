"""Dependency management utility functions."""

# Standard
import functools
import itertools
import operator
from collections.abc import Callable, Collection, Iterable
from typing import Literal, overload

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier

type MembershipFunc = Callable[[Identifier], bool]
"""Function indicating whether an identifier is a member of a given set."""


def is_gather(identifier: Identifier) -> bool:
    """Return whether an identifier is of the 'gather' type."""
    return identifier.is_gather


def is_member(collection: Collection[str] | Collection[Identifier]) -> MembershipFunc:
    """
    Return an identifier membership function.

    Args:
        collection: Collection of node names or identifiers.

    Returns:
        Function that accepts an identifier and returns true when the identifier
            is deemed a member of `collection`, either by node name or by complete
            equality.
    """

    def str_func(identifier: Identifier) -> bool:
        return identifier.node in collection

    def ident_func(identifier: Identifier) -> bool:
        return identifier in collection

    if all(isinstance(obj, str) for obj in collection):
        return str_func
    else:
        return ident_func


def is_ungathered_in(graph: Graph[Identifier]) -> MembershipFunc:
    """
    Return a function that determines whether an identifier is ungathered in a graph.

    Args:
        graph: Identifier graph to search.

    Returns:
        Function that accepts an identifier and returns true when the identifier
            is ungathered in `graph`, i.e. when it is a non-gather node without
            any dependencies.
    """

    def func(identifier: Identifier) -> bool:
        return not identifier.is_gather and not graph[identifier]

    return func


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


def find_separator(
    graph: Graph[Identifier],
    source: MembershipFunc,
    sink: MembershipFunc | None = None,
) -> set[Identifier]:
    """
    Find the separator of two node subsets in an identifier graph.

    Because identifier graphs are directed and bipartite, separators are
    directed from a set of source gather nodes to a set of sink gather nodes.
    This definition of separator is therefore probably a bit non-canonical.

    Args:
        graph: Graph to search.
        source: Membership function of the source set.
        sink: Optional membership function of the sink set, or `None` to use
            the node complement of the source set.

    Returns:
        All nodes in the source set that are depended on by nodes in the sink set.
    """
    source_set = set(filter(source, graph))

    sink_set = set(filter(sink, graph)) if sink else set(graph)
    sink_set -= source_set

    return {
        identifier
        for identifier in source_set
        if any(
            neighbor
            for neighbor in sink_set
            if identifier in graph[neighbor]
        )
    }


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


def find_ungathered(graph: Graph[Identifier]) -> set[Identifier]:
    """
    Find the dependency-less non-gather nodes in an identifier graph.

    Args:
        graph: Graph to search.

    Returns:
        Ungathered nodes. These represent missing inputs to data flow components.
    """
    return set(filter(is_ungathered_in(graph), graph))


def assert_valid(graph: Graph[Identifier], terminal: Identifier):
    """
    Raise an exception if an identifier graph is invalid.

    Args:
        graph: Graph to validate.
        terminal: Terminal node in the graph, which is the only requirement.

    Raises:
        ValueError: If the graph contains unused or ungathered nodes.
    """
    unused = find_unused(graph, subset=(terminal,))
    ungathered = find_ungathered(graph)
    if unused or ungathered:
        raise ValueError(
            "invalid data flow: "
            f"unused components: ({', '.join(unused)}); "
            f"unresolved inputs: ({', '.join(map(str, ungathered))})"
        )


__all__ = [
    "assert_valid",
    "find_separator",
    "find_ungathered",
    "find_unused",
    "is_gather",
    "is_ungathered_in",
    "merge_graphs",
    "sort_gathers",
    "to_node",
]
