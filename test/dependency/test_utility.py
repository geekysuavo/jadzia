"""Tests of dependency utility functions."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.utility import (
    assert_valid,
    find_separator,
    find_ungathered,
    find_unused,
    is_gather,
    is_member,
    merge_graphs,
    sort_gathers,
    to_node,
)


def test_is_gather():
    """Gather identifiers may be identified."""
    assert is_gather(Identifier.gather("foo")) is True
    assert is_gather(Identifier("foo", "bar")) is False


@pytest.mark.parametrize("node", ["foo", "zab"])
def test_to_node(node):
    """Node names may be extracted."""
    identifier = Identifier(node, "out")
    assert to_node(identifier) == node


def test_merge_graphs():
    """Graphs without overlapping gather nodes may be merged."""
    first_deps = {"a.*": (), "a.out": ("a.*",)}
    second_deps = {"b.*": ("a.out",), "b.out": ("b.*",)}
    third_deps = {"c.*": ("b.out",), "c.out": ("c.*",)}
    expected_deps = {
        "a.*": (),
        "a.out": ("a.*",),
        "b.*": ("a.out",),
        "b.out": ("b.*",),
        "c.*": ("b.out",),
        "c.out": ("c.*",),
    }

    first = Graph(first_deps).map(Identifier.from_str)
    second = Graph(second_deps).map(Identifier.from_str)
    third = Graph(third_deps).map(Identifier.from_str)
    expected = Graph(expected_deps).map(Identifier.from_str)

    merged = merge_graphs(first, second, third)
    assert merged == expected

    third.update(Identifier.from_str("b.other"), Identifier.from_str("b.*"))
    with pytest.raises(ValueError, match="names are not unique"):
        merge_graphs(first, second, third)


def test_sort_gathers():
    """Gather nodes may be topologically sorted."""
    deps = {
        "foo.*": (),
        "foo.x": ("foo.*",),
        "foo.y": ("foo.*",),
        "bar.*": ("foo.x",),
        "bar.z": ("bar.*",),
        "baz.*": ("foo.y", "bar.z"),
    }
    graph = Graph(deps).map(Identifier.from_str)

    assert sort_gathers(graph) == ("foo", "bar", "baz")

    subset = (Identifier.from_str("bar.*"),)
    assert sort_gathers(graph, subset) == ("foo", "bar")


@pytest.mark.parametrize("reverse", [False, True])
def test_find_separator(reverse):
    """Separators may be identified in graphs as expected."""
    deps = {"a.*": (), "a.out": ("a.*",), "b.*": ("a.out",), "b.out": ("b.*",)}
    graph = Graph(deps).map(Identifier.from_str)

    source = is_member({"a"})
    sink = is_member({"b"})
    if reverse:
        (source, sink) = (sink, source)

    sepset = find_separator(graph, source, sink)
    if reverse:
        assert sepset == set()
    else:
        assert sepset == {Identifier.from_str("a.out")}


def test_find_unused():
    """Unused gather nodes may be identified by sorting."""
    deps = {
        "foo.*": (),
        "foo.x": ("foo.*",),
        "foo.y": ("foo.*",),
        "bar.*": ("foo.x",),
        "bar.z": ("bar.*",),
        "baz.*": ("foo.y", "bar.z"),
        "baz.w": ("baz.*",),
    }
    graph = Graph(deps).map(Identifier.from_str)

    assert find_unused(graph) == set()
    assert find_unused(graph, gathers=False) == set()

    subset = (Identifier.from_str("bar.z"),)
    assert find_unused(graph, subset) == {"baz"}
    assert find_unused(graph, subset, gathers=False) == {
        Identifier.from_str("foo.y"),
        Identifier.from_str("baz.*"),
        Identifier.from_str("baz.w"),
    }


def test_find_ungathered():
    """Ungathered nodes may be identified by searching."""
    ab = Identifier.gather("ab")
    xy = Identifier.gather("xy")
    ab_out = Identifier.from_str("ab.out")
    xy_out = Identifier.from_str("xy.out")

    graph = Graph[Identifier]()
    graph.update(xy_out)
    assert find_ungathered(graph) == {xy_out}

    graph.update(xy_out, xy)
    assert find_ungathered(graph) == set()

    graph.update(xy, ab_out)
    assert find_ungathered(graph) == {ab_out}

    graph.update(ab_out, ab)
    assert find_ungathered(graph) == set()


@pytest.mark.parametrize("ungathered", [False, True])
@pytest.mark.parametrize("unused", [False, True])
def test_assert_valid(unused, ungathered):
    """Invalid graphs raise exceptions."""
    deps = {
        "source.*": (),
        "source.out": ("source.*",),
        "sink.*": ("source.out",),
        "sink.out": ("sink.*",)
    }

    if unused:
        deps["extra.*"] = ("source.out",)
        deps["extra.out"] = ("extra.*",)

    if ungathered:
        deps["sink.*"] = ("source.out", "ghost.out")
        deps["ghost.out"] = ()

    graph = Graph(deps).map(Identifier.from_str)
    terminal = Identifier.from_str("sink.out")

    valid = not unused and not ungathered
    if valid:
        assert_valid(graph, terminal)
    else:
        with pytest.raises(ValueError, match="invalid data flow"):
            assert_valid(graph, terminal)
