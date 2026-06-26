"""Run configuration base classes."""

# Standard
from enum import StrEnum

# Dax
from jadzia.config.data.loader import DataConfig
from jadzia.config.model import CompositeModel
from jadzia.core.validation.base import Config
from jadzia.core.validation.context import WithRuntimeContext
from jadzia.core.validation.schema import SchemaVersionStr


class Dataset(StrEnum):
    """
    Enumeration of dataset types to include when building data flow graphs.

    This enumeration does not contain `inference`, as there is only
    a single choice for which dataset to build in inference configs.

    - `training`: Include the training dataset.
    - `testing`: Include the testing dataset.
    """

    TRAIN = "training"
    TEST = "testing"


class Group(StrEnum):
    """
    Enumeration of subsets that data flow graphs will be partitioned into.

    - `manifest`: Data manifests.
    - `archive`: Data archives.
    - `transform`: Transformations.
    - `batch`: Batching operations.
    - `model`: Model building blocks.
    - `loss`: Loss function terms.
    - `eval`: Evaluators.
    """

    MANIFEST = "manifest"
    ARCHIVE = "archive"
    TRANSFORM = "transform"
    BATCH = "batch"
    MODEL = "model"
    LOSS = "loss"
    EVAL = "eval"


class RunConfig(Config, WithRuntimeContext):
    """
    Configuration of a `jadzia` run.

    Attributes:
        version: Configuration schema version.
        data: Dataset configurations.
        model: Model configuration.
        loss: Loss function configuration.
        optimizer: Optimizer configuration.
    """

    version: SchemaVersionStr

    data: DataConfig
    model: CompositeModel


__all__ = ["Dataset", "Group", "RunConfig"]
