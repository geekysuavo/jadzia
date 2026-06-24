"""Test configuration and fixtures."""

# Third party
import polars as pl
import pytest

# Dax
from jadzia.data.archive import DataArchive


@pytest.fixture(scope="session")
def session_tmp_path(tmp_path_factory):
    """Directory containing data shared by the entire pytest session."""
    return tmp_path_factory.mktemp("session-shared")


@pytest.fixture(scope="session")
def parquet_file(session_tmp_path):
    """Example file containing `parquet` data, refers into an archive."""
    data = {
        "label": [0.1, 2.3, 4.5, 6.7, 8.9, 1.0],
        "ref": ["foo", "bar"] * 3,
    }
    out_path = session_tmp_path / "manifest.parquet"
    pl.DataFrame(data).write_parquet(out_path)
    return out_path


@pytest.fixture(scope="session")
def parquet_file_no_ref(session_tmp_path):
    """Example file containing `parquet` data, does not refer into an archive."""
    data = {
        "label": [1.5, 3.2, 2.6, 8.7, 4.9],
        "val": [42, 57, 42, 57, 42],
    }
    out_path = session_tmp_path / "manifest-b.parquet"
    pl.DataFrame(data).write_parquet(out_path)
    return out_path


@pytest.fixture(scope="session")
def lmdb_archive_dir(session_tmp_path):
    """Example directory containing `lmdb` archive data."""
    out_path = session_tmp_path / "archive"
    archive = DataArchive(out_path, read_only=False)
    archive["foo"] = 42
    archive["bar"] = 57
    del archive

    return out_path
