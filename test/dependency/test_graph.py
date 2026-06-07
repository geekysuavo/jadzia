"""Tests of dependency graphs."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.graph import Graph


@pytest.mark.parametrize("subset", [None, ("foo",)])
def test_chain(subset):
    """Chain graphs produce correct execution orders."""
    deps = {"foo": ("bar",), "bar": ("baz",), "baz": ("zzz",), "zzz": ()}
    assert Graph(deps).sort_nodes(subset) == ("zzz", "baz", "bar", "foo")


@pytest.mark.parametrize("subset", [None, ("foo",)])
def test_diamond(subset):
    """Diamond graphs produce correct execution orders."""
    deps = {"foo": ("bar", "baz"), "bar": ("zzz",), "baz": ("zzz",), "zzz": ()}
    assert Graph(deps).sort_nodes(subset) == ("zzz", "bar", "baz", "foo")


def test_invalid_subset():
    """Passing invalid subsets raises an exception."""
    deps = {"foo": ("bar",), "bar": ()}
    with pytest.raises(ValueError, match="graph does not contain"):
        Graph(deps).sort_nodes(subset=("baz",))


def test_cycle():
    """Cyclic graphs raise exceptions."""
    deps = {"a": ("b",), "b": ("a",)}
    graph = Graph(deps)
    with pytest.raises(ValueError, match="graph is not a DAG"):
        graph.sort_nodes()


def test_mapping():
    """Dependency graphs are mutable mappings without a handful of methods."""
    graph = Graph()
    assert graph.dependencies == ()
    assert graph.sort_nodes() == ()

    with pytest.raises(KeyError):
        graph["x"]

    graph["a"] = ()
    assert len(graph) == 1
    assert set(graph) == {"a"}
    assert "a" in graph
    assert graph.dependencies == ()
    assert graph.sort_nodes() == ("a",)

    graph.update("a", "b")
    assert len(graph) == 2
    assert set(graph) == {"a", "b"}
    assert "a" in graph
    assert "b" in graph
    assert graph.dependencies == (("a", ("b",)),)
    assert graph.sort_nodes() == ("b", "a")

    equal_graph = Graph({"a": ("b",), "b": ()})
    assert graph == equal_graph
    assert graph != equal_graph.dependencies

    graph["c"] = ("a", "b")
    assert len(graph) == 3
    assert set(graph) == {"a", "b", "c"}
    assert "a" in graph
    assert "b" in graph
    assert "c" in graph
    assert graph.dependencies == (("a", ("b",)), ("c", ("a", "b")))
    assert graph.sort_nodes() == ("b", "a", "c")

    del graph["a"]
    assert len(graph) == 2
    assert set(graph) == {"b", "c"}
    assert "a" not in graph
    assert "b" in graph
    assert "c" in graph
    assert graph.dependencies == (("c", ("b",)),)
    assert graph.sort_nodes() == ("b", "c")

    assert graph != equal_graph
    assert graph != equal_graph.dependencies
