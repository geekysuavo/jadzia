"""Dependency management utility functions."""

# Standard
import functools
import itertools
import operator

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
        ValueError: If any pair of graphs contain identical gather nodes.
    """
    overlaps = set()
    for first, second in itertools.combinations(graphs, 2):
        overlaps.update(filter(is_gather, first & second))

    if overlaps:
        overlap_str = ", ".join(map(str, overlaps))
        raise ValueError(f"names are not unique: {overlap_str}")

    return functools.reduce(operator.or_, graphs)


__all__ = ["is_gather", "merge_graphs", "to_node"]
