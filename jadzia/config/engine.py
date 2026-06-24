"""Engine configuration."""

# Standard
from pathlib import Path

# Third party
from pydantic import DirectoryPath, PositiveInt

# Dax
from jadzia.config.checkpoint import CheckpointConfig
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, IdentifierStr, Reserved
from jadzia.core.validation.base import Config


class IntervalConfig(Config):
    """
    Training interval configuration.

    Attributes:
        logging: How often (in training steps) to log the current train loss.
        testing: How often (in training steps) to log the test losses.
    """

    logging: PositiveInt = 100
    testing: PositiveInt = 1000


class TrainConfig(Config):
    """
    Training loop configuration.

    Attributes:
        dataset: Dataset to use for training.
        steps: Maximum number of training steps.
    """

    dataset: str
    steps: PositiveInt


class TestConfig(Config):
    """
    Testing loop configuration.

    Attributes:
        dataset: Dataset to use for testing.
        steps: Optional maximum number of testing steps. If unset, the entire testing
            dataset will be used.
    """

    dataset: str
    steps: PositiveInt | None = None


class InferConfig(Config):
    """
    Inference loop configuration.

    Attributes:
        dataset: Dataset to use for inference.
        capture: Dataset and model outputs to save.
        checkpoint_dir: Directory to load model state from.
        output_file: File to write captured outputs to in `parquet` format.
            If unset, no output file will be written.
    """

    dataset: str
    capture: tuple[IdentifierStr, ...]
    checkpoint_dir: DirectoryPath
    output_file: Path | None

    def build_subgraph(self) -> tuple[Graph[Identifier], Identifier]:
        """Build a dependency graph based on this configuration."""
        graph = Graph[Identifier]()

        gather = Identifier.gather(Reserved.EVAL)
        terminal = Reserved.EVAL("output")
        graph[gather] = self.capture
        graph[terminal] = (gather,)

        return (graph, terminal)


class TrainingEngineConfig(Config):
    """
    Training engine configuration.

    Attributes:
        intervals: Training interval configuration.
        checkpoints: Checkpointer configuration.
        training: Training loop configuration.
        testing: Testing loop configuration.
    """

    intervals: IntervalConfig = IntervalConfig()
    checkpoints: CheckpointConfig | None = None
    training: TrainConfig
    testing: TestConfig


class InferenceEngineConfig(Config):
    """
    Inference engine configuration.

    Attributes:
        checkpoints: Checkpointer configuration.
        inference: Inference loop configuration.
    """

    checkpoints: CheckpointConfig | None = None
    inference: InferConfig


type EngineConfig = TrainingEngineConfig | InferenceEngineConfig
"""Type alias for all supported engine configurations."""

__all__ = ["EngineConfig", "InferenceEngineConfig", "TrainingEngineConfig"]
