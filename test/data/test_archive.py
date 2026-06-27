"""Tests of data archives."""

# Standard
import pickle

# Third party
import pytest

# Dax
from jadzia.data.archive import DataArchive


def test_archive(tmp_path):
    """Data archives behave as persistent mappings."""
    with pytest.raises(FileNotFoundError):
        DataArchive(tmp_path / "does-not-exist")

    writer = DataArchive(tmp_path, read_only=False)
    assert "x" not in writer
    with pytest.raises(KeyError):
        writer["y"]

    writer.update({"x": 2, "y": 3.3, "z": [5, 7]})
    assert "x" in writer
    assert writer["y"] == 3.3

    assert "xyz" not in writer
    writer["xyz"] = 2.3357
    assert "xyz" in writer
    del writer["xyz"]
    assert "xyz" not in writer

    with pytest.raises(KeyError):
        del writer["xyz"]

    items = tuple(writer.items())
    size = len(writer)
    del writer

    reader = DataArchive(tmp_path)
    assert tuple(reader.items()) == items
    assert len(reader) == size

    (keys, values) = zip(*items)
    assert tuple(reader) == keys
    assert tuple(reader.keys()) == keys
    assert tuple(reader.values()) == values

    with pytest.raises(RuntimeError, match="read-only"):
        reader.update({})

    with pytest.raises(RuntimeError, match="read-only"):
        reader["w"] = 11

    with pytest.raises(RuntimeError, match="read-only"):
        del reader["x"]

    reader_pickled = pickle.dumps(reader)
    del reader
    reader = pickle.loads(reader_pickled)
    assert tuple(reader.items()) == items
