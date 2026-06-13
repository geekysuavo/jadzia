"""Tests of dataset configurations."""

# Third party
import pytest

# Dax
from jadzia.config.data.archive import ArchiveConfig
from jadzia.data.archive import DataArchive


@pytest.fixture
def lmdb_archive_dir(tmp_path):
    """Directory containing an `lmdb` archive."""
    archive = DataArchive(tmp_path, read_only=False)
    archive["foo"] = 42
    archive["bar"] = 57

    return tmp_path


def test_archive_config(lmdb_archive_dir):
    """Archive configurations build cached `lmdb` archives."""
    config = ArchiveConfig(path=lmdb_archive_dir)
    obj = config.build_instance()
    assert isinstance(obj, DataArchive)
    assert dict(obj.items()) == {"foo": 42, "bar": 57}


def test_archive_config_caching(mocker, tmp_path):
    """Built data archives are cached by path."""
    mock_archive = mocker.patch("jadzia.config.data.archive.DataArchive")
    ArchiveConfig.clear_cache()

    config = ArchiveConfig(path=tmp_path)
    config.build_instance()
    mock_archive.assert_called_once()
    mock_archive.reset_mock()

    config.build_instance()
    mock_archive.assert_not_called()

    ArchiveConfig.clear_cache()
    config.build_instance()
    mock_archive.assert_called_once()
