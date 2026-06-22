"""Checkpointing functionality."""

# Standard
from typing import NamedTuple, Self

# Third party
import equinox as eqx
import grain
import optax
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.nn.model import CompositeModel


class Checkpointables(NamedTuple):
    """Types of each checkpointable stored within a checkpoint."""

    step: int
    model: CompositeModel
    model_state: eqx.nn.State
    dataloader_iter: grain.DataLoaderIterator
    optimizer_state: optax.OptState

    def save(self, checkpointer: ocp.training.Checkpointer | None) -> bool:
        """
        Save checkpointable data.

        Args:
            checkpointer: Optional checkpointer used to save data.

        Returns:
            Whether a checkpoint was saved.
        """
        if checkpointer is None:
            return False

        return checkpointer.save_checkpointables(
            step=self.step,
            checkpointables=self._asdict(),
        )

    def load(self, checkpointer: ocp.training.Checkpointer | None) -> Self | None:
        """
        Load checkpointable data.

        Args:
            checkpointer: Optional checkpointer used to load data.

        Returns:
            Loaded data, or `None` if no data were loaded.
        """
        if checkpointer is None:
            return None

        if checkpointer.latest is None:
            return None

        loaded = checkpointer.load_checkpointables(
            abstract_checkpointables=self._asdict(),
        )
        return type(self)(**loaded)


__all__ = ["Checkpointables"]
