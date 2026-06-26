"""Tests of identifier graph validation functions."""

# Third party
import pytest

# Dax
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.dataflow import validate_data_flow


@pytest.fixture
def deps():
    return {
        "src.*": set(),
        "src.feature": {"src.*"},
        "src.label": {"src.*"},
        "src.extra": {"src.*"},
        "src.skip": {"src.*"},
        "foo.*": {"src.feature"},
        "foo.out": {"foo.*"},
        "foo.extra": {"foo.*"},
        "bar.*": {"src.label"},
        "bar.out": {"bar.*"},
        "bar.extra": {"bar.*"},
        "sink.*": {"foo.out", "bar.out", "src.skip"},
        "sink.out": {"sink.*"},
    }


def validate(deps):
    graph = Graph(deps).map(Identifier.from_str)
    terminal = Identifier.from_str("sink.out")
    assignment = {
        "src": ("src",),
        "pipe": ("foo", "bar", "baz"),
        "sink": ("sink",),
    }
    allowed = {"pipe": ("src",), "sink": ("src", "pipe")}
    extra_reductions = {"skip": (("src", "sink"),)}
    return validate_data_flow(
        graph,
        terminal,
        assignment=assignment,
        allowed=allowed,
        extra_reductions=extra_reductions,
    )


def test_valid(deps):
    """Valid graphs return a dictionary of outputs from each assigned group."""
    sepsets = validate(deps)
    assert sepsets == {
        "src": {
            Identifier("src", "feature"),
            Identifier("src", "label"),
            Identifier("src", "skip"),
        },
        "pipe": {Identifier("foo", "out"), Identifier("bar", "out")},
        "sink": set(),
        "skip": {Identifier("src", "skip")},
    }


@pytest.mark.parametrize("node", ["sink.out", "foo.extra", "bar.out"])
def test_cycle(deps, node):
    """Graphs with cycles raise an exception."""
    deps["src.*"].add(node)
    with pytest.raises(ValueError, match="not a DAG"):
        validate(deps)


def test_unused(deps):
    """Graphs with unused gather nodes raise an exception."""
    deps["baz.*"] = {"src.label"}
    deps["baz.out"] = {"baz.*"}
    with pytest.raises(ValueError, match="unused"):
        validate(deps)


def test_ungathered(deps):
    """Graphs with ungathered nodes raise an exception."""
    deps["src.oops"] = set()
    deps["bar.*"].add("src.oops")
    with pytest.raises(ValueError, match="unresolved"):
        validate(deps)


def test_disallowed(deps):
    """Graphs with disallowed data flow raise an exception."""
    deps["sink.*"] = set()
    deps["sink.other"] = {"sink.*"}
    deps["baz.*"] = {"sink.other"}
    deps["sink.out"] = {"baz.*"}
    with pytest.raises(ValueError, match=r"disallowed.*(sink->pipe)"):
        validate(deps)
