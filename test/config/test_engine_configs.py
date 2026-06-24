"""Tests of engine configurations."""

# Dax
from jadzia.config.engine import InferConfig


def test_infer_config(tmp_path):
    """The inference config builds graphs of the expected structure."""
    config = InferConfig(
        dataset="test",
        capture=("foo.bar", "hello.world"),
        checkpoint_dir=tmp_path,
        output_file=None,
    )

    (graph, terminal) = config.build_subgraph()
    assert all(node in graph for node in config.capture)
    assert terminal in graph

    order = graph.sort_nodes(subset=(terminal,))
    assert all(node in order for node in config.capture)
