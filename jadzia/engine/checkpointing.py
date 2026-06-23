"""Checkpointing functionality."""

# Standard
from functools import partial
from pathlib import Path
from typing import Self, overload

# Third party
import equinox as eqx
import grain
import optax
from orbax.checkpoint import v1 as ocp

# Dax
from jadzia.core.meta import AbstractDataclassMeta
from jadzia.nn.model import CompositeModel


class Checkpointables(metaclass=AbstractDataclassMeta):
    """Base class for loading and saving checkpointables."""

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
            step=getattr(self, "step", 0),
            checkpointables=self.__dict__.copy(),
        )

    @overload
    def load(self, checkpointer: ocp.training.Checkpointer) -> Self | None: ...

    @overload
    def load(self, checkpointer: Path) -> Self: ...

    @overload
    def load(self, checkpointer: None) -> None: ...

    def load(
        self,
        checkpointer: ocp.training.Checkpointer | Path | None,
    ) -> Self | None:
        """
        Load checkpointable data.

        Args:
            checkpointer: Optional checkpointer used to load data, or a specific
                directory to load checkpoint data from.

        Returns:
            Loaded data, or `None` if no data were loaded.
        """
        if isinstance(checkpointer, Path):
            if checkpointer.is_dir():
                load = partial(ocp.load_checkpointables, str(checkpointer))
            else:
                raise NotADirectoryError(str(checkpointer))

        elif isinstance(checkpointer, ocp.training.Checkpointer):
            if checkpointer.latest is not None:
                load = checkpointer.load_checkpointables
            else:
                return None

        elif checkpointer is None:
            return None

        loaded = load(abstract_checkpointables=self.__dict__.copy())
        return type(self)(**loaded)


class Checkpoint(Checkpointables):
    """Types of each checkpointable stored within a full checkpoint."""

    step: int
    model: CompositeModel
    model_state: eqx.nn.State
    dataloader_iter: grain.DataLoaderIterator
    optimizer_state: optax.OptState


class ModelCheckpoint(Checkpointables):
    """Checkpointables relating only to the model and its state."""

    model: CompositeModel
    model_state: eqx.nn.State


class DataLoaderCheckpoint(Checkpointables):
    """Checkpointables relating only to the data loader iterator."""

    step: int
    dataloader_iter: grain.DataLoaderIterator


__all__ = ["Checkpoint", "DataLoaderCheckpoint", "ModelCheckpoint"]
