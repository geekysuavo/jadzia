"""Tests of dependency traces."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.trace import Trace


def test_naming():
    """Node names are determined based on their object."""
    assert str(Trace("hello")) == "hello"
    assert str(Trace({})) == "dict"


def test_initialization():
    """Names are converted to sets. Empty values and `None` are removed."""
    root = Trace("root", inserts=["a", "a", "b", "", None])

    assert "a" in root
    assert "b" in root

    assert None not in root
    assert "c" not in root
    assert "" not in root

    assert set(root) == {"a", "b"}
    assert sorted(root) == ["a", "b"]


def test_root_finding():
    """Root nodes may be found from any part of the tree."""
    root = Trace("root")
    assert root.root() is root

    left = root.update("left")
    assert left.root() is root

    right = root.update("right")
    assert right.root() is root

    right_child = right.update("child")
    assert right_child.root() is root


def test_updating():
    """The available name set is built as the tree is constructed."""
    root = Trace("root", inserts="abc")
    assert set(root) == {"a", "b", "c"}

    def check_foo(names):
        assert names == {"a", "b"}

    node = root.update("foo", requires="ab", inserts="d", callback=check_foo)
    assert set(node) == {"a", "b", "c", "d"}

    def check_bar(names):
        assert names == {"a"}

    node = node.update("bar", requires="d", removes="a", callback=check_bar)
    assert set(node) == {"b", "c", "d"}

    assert node.root().traverse() == {"a", "b"}


def test_checking():
    """Input requirements are checked as the tree is constructed."""
    root = Trace("root", inserts={"some"})

    with pytest.raises(ValueError, match="failed to trace data flow"):
        root.update("requires-missing", requires={"missing"})

    with pytest.raises(ValueError, match="failed to trace data flow"):
        root.update("removes-missing", removes={"another-missing"})

    with pytest.raises(ValueError, match="root nodes may not have"):
        Trace("root-cannot-require", requires={"foo"})

    with pytest.raises(ValueError, match="root nodes may not have"):
        Trace("root-cannot-remove", removes={"foo"})
