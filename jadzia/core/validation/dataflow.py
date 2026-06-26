"""Identifier graph validation functions."""

# Standard
import itertools
from collections.abc import Collection, Iterable

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.utility import (
    find_separator,
    find_ungathered,
    find_unused,
    is_member,
)


def validate_data_flow(
    graph: Graph[Identifier],
    terminal: Identifier,
    *,
    assignment: dict[str, Collection[str] | Collection[Identifier]],
    allowed: dict[str, Iterable[str]],
    extra_reductions: dict[str, Iterable[tuple[str, str]]],
) -> dict[str, set[Identifier]]:
    """
    Validate an identifier graph and return per-component output sets.

    Args:
        graph: Graph to validate.
        terminal: Terminal node in the graph, which is the only requirement.
        assignment: Named collections of node names.
        allowed: Dictionary of source collections allowed by each sink collection.
        extra_reductions: Additional reductions to perform and include in the
            returned dictionary of separators.

    Returns:
        Minimum required outputs for each collection in the `assignment`.

    Raises:
        ValueError: If the graph contains unused or ungathered nodes,
            or if any disallowed separators are non-empty.
    """
    membership = {name: is_member(assn) for name, assn in assignment.items()}
    separators = {
        (source, sink): find_separator(graph, membership[source], membership[sink])
        for source, sink in itertools.permutations(membership, 2)
    }

    allowed_set = {
        (source, sink)
        for sink, allowed_sources in allowed.items()
        for source in allowed_sources
    }

    disallowed = {
        f"({source}->{sink}): ({', '.join(map(str, sepset))})"
        for (source, sink), sepset in separators.items()
        if sepset and (source, sink) not in allowed_set
    }

    unused = find_unused(graph, subset=(terminal,))
    ungathered = find_ungathered(graph)

    if disallowed or unused or ungathered:
        raise ValueError(
            "invalid data flow: "
            f"unused components: ({', '.join(unused)}); "
            f"unresolved inputs: ({', '.join(map(str, ungathered))}); "
            f"disallowed data flow: {', '.join(disallowed)}"
        )

    result = {}
    for (source, sink), sepset in separators.items():
        result.setdefault(source, set()).update(sepset)

    for key, pairs in extra_reductions.items():
        for source, sink in pairs:
            sepset = separators[(source, sink)]
            result.setdefault(key, set()).update(sepset)

    return result


__all__ = ["validate_data_flow"]
