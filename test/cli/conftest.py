"""CLI test fixtures."""

# Standard
import random
from pathlib import Path

# Third party
import numpy as np
import polars as pl
import pytest

# Dax
from jadzia.data.archive import DataArchive

num_rows: int = 1000
ref_values: str = "abcdefghijklmnopqrstuvwxyz"


@pytest.fixture
def parquet_file(tmp_path):
    """Example file containing `parquet` data."""
    data = {
        "label": [random.normalvariate() for _ in range(num_rows)],
        "ref": [random.choice(ref_values) for _ in range(num_rows)],
    }
    out_file = tmp_path / "data.parquet"
    pl.DataFrame(data).write_parquet(out_file)
    return out_file


@pytest.fixture
def lmdb_archive_dir(tmp_path):
    """Example directory containing `lmdb` archive data."""
    archive = DataArchive(tmp_path, read_only=False)
    rng = np.random.default_rng()
    for key in ref_values:
        archive[key] = rng.normal(size=(32,))

    del archive
    return tmp_path


@pytest.fixture
def write_train_config_file(tmp_path, parquet_file, lmdb_archive_dir):
    """Training configuration file writer."""

    def writer() -> Path:
        source_file = Path(__file__).parent / "config-train.yaml"
        dest_file = tmp_path / source_file.name
        content = (
            source_file.read_text()
            .replace("!PARQUET_FILE", str(parquet_file))
            .replace("!LMDB_ARCHIVE_DIR", str(lmdb_archive_dir))
        )
        dest_file.write_text(content)
        return dest_file

    return writer


@pytest.fixture
def write_infer_config_file(tmp_path, parquet_file, lmdb_archive_dir):
    """Inference configuration file writer."""

    def writer(checkpoint_dir: Path) -> Path:
        source_file = Path(__file__).parent / "config-infer.yaml"
        dest_file = tmp_path / source_file.name
        out_file = tmp_path / "inferences.parquet"
        content = (
            source_file.read_text()
            .replace("!PARQUET_FILE", str(parquet_file))
            .replace("!LMDB_ARCHIVE_DIR", str(lmdb_archive_dir))
            .replace("!CHECKPOINT_DIR", str(checkpoint_dir))
            .replace("!OUTPUT_FILE", str(out_file))
        )
        dest_file.write_text(content)
        return dest_file

    return writer
