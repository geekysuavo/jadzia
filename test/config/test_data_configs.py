"""Tests of dataset configurations."""

# Third party
import grain
import polars as pl
import pytest

# Dax
from jadzia.config.data.archive import ArchiveConfig
from jadzia.config.data.loader import DataLoaderConfig
from jadzia.config.data.option import ReaderConfig, ShardingConfig
from jadzia.config.data.source import SourceConfig, SourceMappingConfig
from jadzia.config.data.transform import BatchConfig, CompositeTransform
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.context import RuntimeContext
from jadzia.data.archive import DataArchive
from jadzia.data.sharedmemory import SharedDataFrame
from jadzia.data.source import DataSource
from jadzia.data.transform.special import ConvertNone, SelectKeys


@pytest.fixture
def parquet_file(tmp_path):
    """File containing `parquet` data."""
    data = {
        "label": [0.1, 2.3, 4.5, 6.7, 8.9, 1.0],
        "ref": ["foo", "bar"] * 3,
    }
    out_path = tmp_path / "manifest.parquet"
    pl.DataFrame(data).write_parquet(out_path)
    return out_path


@pytest.fixture
def parquet_file_no_ref(tmp_path):
    """File containing `parquet` data that does not refer into an archive."""
    data = {
        "label": [1.5, 3.2, 2.6, 8.7, 4.9],
        "val": [42, 57, 42, 57, 42],
    }
    out_path = tmp_path / "manifest-b.parquet"
    pl.DataFrame(data).write_parquet(out_path)
    return out_path


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


def test_source_config(parquet_file, lmdb_archive_dir):
    """Source configurations build cached shared data frames with attached archives."""
    config = SourceConfig(
        manifest=parquet_file,
        archives={"ref": ArchiveConfig(path=lmdb_archive_dir)},
    )

    df = config.read()
    assert isinstance(df, pl.DataFrame)

    shared_df = config.share(df)
    assert isinstance(shared_df, SharedDataFrame)
    assert config.share(df) is shared_df


@pytest.mark.parametrize("outputs", [None, {"source.val"}], ids=["all", "one"])
def test_source_mapping(parquet_file, parquet_file_no_ref, lmdb_archive_dir, outputs):
    """Source configuration mappings build data sources."""
    config = SourceMappingConfig(
        first=SourceConfig(
            manifest=parquet_file,
            archives={"val": ArchiveConfig(path=lmdb_archive_dir, from_key="ref")},
        ),
        second=SourceConfig(
            manifest=parquet_file_no_ref,
            archives={},
        ),
    )

    first_deps = {
        "source.*": (),
        "id.manifest": ("source.*",),
        "source.label": ("source.*",),
        "source.ref": ("source.*",),
        "source.val": ("source.*",),
    }
    first_graph = config["first"].build_subgraph()
    assert first_graph == Graph(first_deps).map(Identifier.from_str)

    second_deps = first_deps.copy()
    second_deps.pop("source.ref")
    second_graph = config["second"].build_subgraph()
    assert second_graph == Graph(second_deps).map(Identifier.from_str)

    graph = config.build_subgraph()
    assert graph == (first_graph | second_graph)

    dfs = config.build_instance(outputs=outputs)
    source = config.build_instance(dfs, outputs=outputs)
    assert isinstance(source, DataSource)
    if outputs:
        assert set(source[0]) == outputs
    else:
        all_outputs = {"id.manifest", "source.label", "source.ref", "source.val"}
        assert set(source[0]) == all_outputs


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


def test_source_config_caching(mocker, parquet_file):
    """Shared data frames are cached by hash."""
    mock_shared_df = mocker.patch("jadzia.config.data.source.SharedDataFrame")
    SourceConfig.clear_cache()

    config = SourceConfig(manifest=parquet_file, archives={})
    df = config.read()
    mock_shared_df.assert_not_called()

    config.share(df)
    mock_shared_df.assert_called_once()
    mock_shared_df.reset_mock()

    config.share(df)
    mock_shared_df.assert_not_called()

    SourceConfig.clear_cache()
    config.share(df)
    mock_shared_df.assert_called_once()


@pytest.mark.parametrize("outputs", [None, {"a.b"}], ids=[False, True])
@pytest.mark.parametrize("concatenate", [None, {"a.b"}], ids=[False, True])
@pytest.mark.parametrize("convert_none", [True, False])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("size", [1, 8])
def test_batch_config(size, drop, convert_none, concatenate, outputs):
    """Batch transform configuations build `grain.transforms.Batch` instances."""
    config = BatchConfig(
        size=size,
        drop=drop,
        concatenate=concatenate,
        convert_none=convert_none,
    )
    transforms = config.build_instance(outputs)
    assert isinstance(transforms, tuple)

    expected_num = int(convert_none) + int(outputs is not None) + 1
    assert len(transforms) == expected_num

    if convert_none:
        (convert_transform, *transforms) = transforms
        assert isinstance(convert_transform, ConvertNone)

    if outputs:
        (select_transform, *transforms) = transforms
        assert isinstance(select_transform, SelectKeys)

    (*transforms, batch_transform) = transforms
    assert isinstance(batch_transform, grain.transforms.Batch)
    assert batch_transform.batch_size == size
    assert batch_transform.drop_remainder == drop
    assert transforms == []

    deps = {"a.b": (), "batch.*": ("a.b",)} if concatenate else {}
    graph = config.build_subgraph()
    assert graph == Graph(deps).map(Identifier.from_str)


@pytest.mark.parametrize("outputs", [False, True])
@pytest.mark.parametrize("dag", [True, False])
@pytest.mark.parametrize("unused_transform", [False, True])
def test_composite_tform(unused_transform, dag, outputs):
    """Composite transform configurations build sequences of transforms."""
    config_data = {
        "tform_in": {
            "type": "Unary",
            "function": "math.sin",
            "input": "source.label",
        },
        "tform_out": {
            "type": "Unary",
            "function": "math.sin",
            "input": "tform_in.output",
        },
        "tform_extra": {
            "type": "Unary",
            "function": "math.sin",
            "input": "tform_out.output",
        },
        "tform_cycle_a": {
            "type": "Unary",
            "function": "math.sin",
            "input": "tform_cycle_b.output",
        },
        "tform_cycle_b": {
            "type": "Unary",
            "function": "math.sin",
            "input": "tform_cycle_a.output",
        },
    }

    outputs = (
        Identifier("tform_out", "output"),
        Identifier("tform_cycle_b", "output"),
    ) if outputs else None

    if not unused_transform:
        del config_data["tform_extra"]

    if dag:
        outputs = outputs[:1] if outputs else None
        del config_data["tform_cycle_a"]
        del config_data["tform_cycle_b"]

    config = CompositeTransform.model_validate(config_data)

    if not dag:
        with pytest.raises(ValueError, match="graph is not a DAG"):
            config.build_instance(outputs)
        return

    transforms = config.build_instance(outputs)
    assert isinstance(transforms, tuple)
    assert len(transforms) == 2 if outputs else 3


def test_loader_config(parquet_file, lmdb_archive_dir):
    """Data loader configurations build data loaders."""
    config_data = {
        "source": {
            "first": {
                "manifest": parquet_file,
                "archives": {"ref": {"path": lmdb_archive_dir}},
            },
        },
        "transformations": {
            "exp": {"type": "Unary", "input": "source.label", "function": "math.exp"}
        },
        "batch": {"size": 8},
        "sampler": {"type": "Sequential"},
    }
    context = RuntimeContext(name="test", path=parquet_file.parent)
    config = DataLoaderConfig.model_validate(config_data, context=context)

    deps = {
        "source.*": (),
        "id.manifest": ("source.*",),
        "source.label": ("source.*",),
        "source.ref": ("source.*",),
        "exp.*": ("source.label",),
        "exp.output": ("exp.*",),
    }
    graph = config.build_subgraph()
    assert graph == Graph(deps).map(Identifier.from_str)

    loader = config.build_instance()
    assert isinstance(loader, grain.DataLoader)

    batch = next(iter(loader))
    assert isinstance(batch, dict)
