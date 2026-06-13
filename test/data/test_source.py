"""Tests of data sources."""

# Standard
import random

# Third party
import polars as pl
import pytest

# Dax
from jadzia.data.archive import DataArchive
from jadzia.data.sharedmemory import SharedDataFrame
from jadzia.data.source import ArchiveMapping, DataSource


@pytest.mark.parametrize("emplace", [True, False])
@pytest.mark.parametrize("insert_id", [True, False])
@pytest.mark.parametrize("outputs", [None, "ix", "ixyz"])
def test_data_source(tmp_path, outputs, insert_id, emplace):
    """Data sources provide the expected interface and behavior."""
    num_rows = (10, 8)
    data_a = {
        "i": list(range(num_rows[0])),
        "x": [random.uniform(0, 1) for _ in range(num_rows[0])],
        "y": [f"k{index}" for index in range(num_rows[0])],
    }
    data_b = {
        "i": list(range(num_rows[0], sum(num_rows))),
        "x": [random.uniform(0, 1) for _ in range(num_rows[1])],
        "y": [f"k{index + num_rows[0]}" for index in range(num_rows[1])],
    }
    df_a = SharedDataFrame(pl.DataFrame(data_a), create=True)
    df_b = SharedDataFrame(pl.DataFrame(data_b), create=True)
    del data_a, data_b

    archive = DataArchive(tmp_path / "archive", read_only=False)
    data = {f"k{index}": random.randint(0, 100) for index in range(sum(num_rows))}
    archive.update(data)
    del archive, data

    archive = DataArchive(tmp_path / "archive")

    mapping = ArchiveMapping(
        archive="arch",
        source_key="y",
        dest_key="y" if emplace else "z",
    )

    outputs = set(outputs) if outputs else None

    source = DataSource(
        dfs={"alpha": df_a, "beta": df_b},
        archives={"arch": archive},
        mappings={"alpha": [mapping], "beta": [mapping]},
        insert_id=insert_id,
        outputs=outputs,
    )

    assert len(source) == sum(num_rows)

    expected_keys = set("ixy") if emplace else set("ixyz")
    if insert_id:
        expected_keys.add("id.manifest")
    if outputs is not None:
        expected_keys = outputs

    for index in range(len(source)):
        row = source[index]
        assert set(row) == expected_keys
        assert row["i"] == index
        assert isinstance(row["x"], float)

        if outputs is not None:
            continue  # the below keys won't exist. no need to check them.

        if insert_id:
            assert row["id.manifest"] == "alpha" if index < num_rows[0] else "beta"

        if emplace:
            assert isinstance(row["y"], int)
        else:
            assert isinstance(row["y"], str)
            assert isinstance(row["z"], int)

    assert source[-1] == source[sum(num_rows) - 1]
    with pytest.raises(IndexError, match="out of range"):
        source[sum(num_rows)]
