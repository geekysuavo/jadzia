"""Tests of dataset configurations."""

# Third party
import grain
import pytest

# Dax
from jadzia.config.data.archive import ArchiveConfig
from jadzia.config.data.option import ReaderConfig, ShardingConfig
from jadzia.data.archive import DataArchive


@pytest.fixture
def lmdb_archive_dir(tmp_path):
    """Directory containing an `lmdb` archive."""
    archive = DataArchive(tmp_path, read_only=False)
    archive["foo"] = 42
    archive["bar"] = 57

    return tmp_path


@pytest.mark.parametrize("buffer_size", [1, 10])
@pytest.mark.parametrize("num_threads", [0, 6])
def test_reader_config(num_threads, buffer_size):
    """Reader configurations build `grain.ReadOptions` instances."""
    config = ReaderConfig(num_threads=num_threads, buffer_size=buffer_size)
    obj = config.build_instance()
    assert isinstance(obj, grain.ReadOptions)
    assert obj.num_threads == num_threads
    assert obj.prefetch_buffer_size == buffer_size


@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize(("process_count", "process_index"), [(1, 0), (2, 0), (2, 1)])
def test_sharding_config(mocker, process_count, process_index, drop):
    """Sharding configurations build `grain.sharding.ShardOptions` instances."""
    mocker.patch(
        "jadzia.config.data.option.jax.process_count",
        return_value=process_count,
    )
    mocker.patch(
        "jadzia.config.data.option.jax.process_index",
        return_value=process_index,
    )

    config = ShardingConfig(drop=drop)
    obj = config.build_instance()
    if process_count == 1:
        assert isinstance(obj, grain.sharding.NoSharding)
    else:
        assert isinstance(obj, grain.sharding.ShardByJaxProcess)
        assert obj.drop_remainder == drop


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
