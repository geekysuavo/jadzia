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


@pytest.mark.parametrize("subset", [None, ("a",), ("b",)])
def test_tree(subset):
    """Passing valid subsets returns valid orders."""
    deps = {"root": (), "a": ("root",), "b": ("root",)}
    order = Graph(deps).sort_nodes(subset)
    if subset is None:
        assert set(order) == {"a", "b", "root"}
    else:
        assert set(order) == {*subset, "root"}


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


def test_copy():
    """Graphs can be copied."""
    graph = Graph({"a": (), "b": "a", "c": "a"})
    graph_copy = graph.copy()
    assert graph_copy == graph
    assert graph_copy is not graph

    graph_copy.update("d", "c")
    assert graph_copy != graph


def test_union():
    """The bitwise `or` of two graphs returns their union."""
    first = Graph({"a.foo": ("a",), "a.bar": ("a",), "a": ()})
    second = Graph({"a.foo": (), "b": ("a.foo",)})
    union = first | second

    expected = Graph({"a.foo": ("a",), "a.bar": ("a",), "a": (), "b": ("a.foo",)})
    assert union == expected


def test_intersection():
    """The bitwise `and` of two graphs returns their intersection."""
    first = Graph({"a.foo": ("a",), "a.bar": ("a",), "a": ()})
    second = Graph({"a.foo": (), "b": ("a.foo",)})
    intersection = first & second

    expected = Graph({"a.foo": ()})
    assert intersection == expected
