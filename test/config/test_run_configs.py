"""Tests of run configurations."""

# Standard
from copy import deepcopy

# Third party
import pytest

# Dax
from jadzia.config.run.inference import InferenceRunConfig
from jadzia.config.run.training import TrainingRunConfig
from jadzia.core.validation.context import RuntimeContext
from jadzia.engine.inference import InferArgs
from jadzia.engine.training import TrainArgs


@pytest.fixture(params=(True, False))
def config_data(request, parquet_file_no_ref):
    """Example run configuration data."""
    return {
        "version": "0.1",
        "data": {
            "train-test": {
                "source": {
                    "mini": {
                        "manifest": parquet_file_no_ref,
                        "archives": {},
                    },
                },
                "transformations": {},
                "batch": {"size": 2},
                "sampler": {"type": "General"},
            },
        },
        "model": {
            "proj": {
                "type": "Linear",
                "input": "source.val",
                "num_input_features": 1,
                "num_output_features": 1,
            }
        },
        "loss": {
            "mse": {
                "type": "MSE",
                "inputs": {"yhat": "proj.output", "y": "source.label"},
            },
        },
        "optimizer": {"type": "Adam", "learning_rate": 0.001},
        "engine": {
            "checkpoints": {} if request.param else None,
            "training": {"dataset": "train-test", "steps": 1},
            "testing": {"dataset": "train-test"},
            "inference": {
                "dataset": "train-test",
                "capture": ["source.label", "proj.output"],
                "checkpoint_dir": parquet_file_no_ref.parent,
                "output_file": None,
            },
        },
    }


@pytest.fixture
def train_data(config_data):
    """Example training run configuration data."""
    del config_data["engine"]["inference"]
    return config_data


@pytest.fixture
def infer_data(config_data):
    """Example inference run configuration data."""
    del config_data["loss"]
    del config_data["optimizer"]
    del config_data["engine"]["training"]
    del config_data["engine"]["testing"]
    return config_data


def test_training(tmp_path, train_data):
    """The training config validates data flow and builds training args."""
    context = RuntimeContext.from_dir(tmp_path)
    config = TrainingRunConfig.model_validate(train_data, context=context)
    args = config.build_instance()
    assert isinstance(args, TrainArgs)


def test_inference(tmp_path, infer_data):
    """The inference config validates data flow and builds inference args."""
    context = RuntimeContext.from_dir(tmp_path)
    config = InferenceRunConfig.model_validate(infer_data, context=context)
    args = config.build_instance()
    assert isinstance(args, InferArgs)


def test_training_missing_dataset(tmp_path, train_data):
    """Missing datasets raise a validation error."""
    train_data["data"] = {"oops": train_data["data"]["train-test"]}

    context = RuntimeContext.from_dir(tmp_path)
    with pytest.raises(ValueError, match="datasets are missing"):
        TrainingRunConfig.model_validate(train_data, context=context)


def test_inference_missing_dataset(tmp_path, infer_data):
    """Missing datasets raise a validation error."""
    infer_data["data"] = {"oops": infer_data["data"]["train-test"]}

    context = RuntimeContext.from_dir(tmp_path)
    with pytest.raises(ValueError, match="datasets are missing"):
        InferenceRunConfig.model_validate(infer_data, context=context)


def test_training_unused_dataset(tmp_path, train_data):
    """Missing datasets raise a validation error."""
    train_data["data"]["oops"] = deepcopy(train_data["data"]["train-test"])

    context = RuntimeContext.from_dir(tmp_path)
    with pytest.raises(ValueError, match="datasets are unused"):
        TrainingRunConfig.model_validate(train_data, context=context)


def test_inference_unused_dataset(tmp_path, infer_data):
    """Missing datasets raise a validation error."""
    infer_data["data"]["oops"] = deepcopy(infer_data["data"]["train-test"])

    context = RuntimeContext.from_dir(tmp_path)
    with pytest.raises(ValueError, match="datasets are unused"):
        InferenceRunConfig.model_validate(infer_data, context=context)
