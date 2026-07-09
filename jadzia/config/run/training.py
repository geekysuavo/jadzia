"""Training run configuration."""

# Standard
import itertools
from typing import Self

# Third party
import jax.random as jr
from pydantic import PrivateAttr, model_validator

# Dax
from jadzia.config.engine import TrainingEngineConfig
from jadzia.config.loss.composite import CompositeLoss
from jadzia.config.optimizer import Optimizer
from jadzia.config.run.base import Dataset, Group, RunConfig
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.dataflow import validate_data_flow
from jadzia.engine.testing import TestArgs
from jadzia.engine.training import TrainArgs


class TrainingRunConfig(RunConfig):
    """
    Configuration of a `jadzia` training run.

    Attributes:
        version: Configuration schema version.
        data: Dataset configurations.
        model: Model configuration.
        loss: Loss function configuration.
        optimizer: Optimizer configuration.
        engine: Training and testing configuration.
    """

    loss: CompositeLoss
    optimizer: Optimizer
    engine: TrainingEngineConfig

    _separators: dict[str, dict[str, set[Identifier]]] = PrivateAttr({})

    def build_instance(self) -> TrainArgs:
        """Build an instance from this configuration."""
        dataloaders = {}
        for dataset in Dataset:
            manifest_outputs = self._separators[Dataset.TRAIN][Group.MANIFEST]
            archive_outputs = self._separators[Dataset.TRAIN][Group.ARCHIVE]
            batch_outputs = self._separators[Dataset.TRAIN][Group.BATCH]
            model_outputs = self._separators[Dataset.TRAIN][Group.MODEL]

            dataset_name = self._dataset_name(dataset)
            dataloader = self.data[dataset_name].build_instance(
                source_outputs=manifest_outputs | archive_outputs,
                batch_outputs=batch_outputs,
            )
            dataloaders[dataset] = dataloader

        (model, model_state) = self.model.build_instance(
            jr.key(self.context.seed) if self.model.requires_key else None,
            outputs=model_outputs,
            with_state=True,
        )

        loss_fn = self.loss.build_instance()

        optimizer = self.optimizer.build_instance()
        optimizer_state = self.optimizer.init_state(optimizer, model)

        if self.engine.checkpoints is not None:
            checkpointer = self.engine.checkpoints.build_instance()
        else:
            checkpointer = None

        return TrainArgs(
            model=model,
            model_state=model_state,
            loss_fn=loss_fn,
            optimizer=optimizer,
            optimizer_state=optimizer_state,
            dataloader=dataloaders[Dataset.TRAIN],
            sharding=self.context.data_sharding,
            num_steps=self.engine.training.steps,
            checkpointer=checkpointer,
            test_args=TestArgs(
                model=model,
                model_state=model_state,
                loss_fn=loss_fn,
                dataloader=dataloaders[Dataset.TEST],
                sharding=self.context.data_sharding,
                num_steps=self.engine.testing.steps,
            ),
            test_every=self.engine.intervals.testing,
            log_every=self.engine.intervals.logging,
        )

    def build_subgraph(self, dataset: Dataset) -> tuple[Graph[Identifier], Identifier]:
        """
        Build a dependency graph based on this configuration.

        Args:
            dataset: Selection of which dataset to include in the graph.

        Returns:
            Newly built dependency graph.
            Terminal node of the dependency graph.
        """
        dataset_name = self._dataset_name(dataset)
        data_graph = self.data[dataset_name].build_subgraph()
        model_graph = self.model.build_subgraph()
        (loss_graph, terminal) = self.loss.build_subgraph()
        return (Graph.union(data_graph, model_graph, loss_graph), terminal)

    def _dataset_name(self, dataset: Dataset) -> str:
        """Return the name of a dataset used by a given data flow."""
        if dataset == Dataset.TRAIN:
            return self.engine.training.dataset
        elif dataset == Dataset.TEST:
            return self.engine.testing.dataset
        else:
            raise ValueError(f"invalid dataset '{dataset}'")  # pragma: no cover

    @model_validator(mode="after")
    def _validate_data_flow(self) -> Self:
        """Validate the data flow graph."""
        required = {self.engine.training.dataset, self.engine.testing.dataset}
        if missing := required - set(self.data):
            raise ValueError(f"datasets are missing: {', '.join(missing)}")
        elif extra := set(self.data) - required:
            raise ValueError(f"datasets are unused: {', '.join(extra)}")

        for dataset in Dataset:
            dataset_name = self._dataset_name(dataset)
            config = self.data[dataset_name]

            self._separators[dataset] = validate_data_flow(
                *self.build_subgraph(dataset),
                assignment={
                    Group.MANIFEST: config.source,
                    Group.ARCHIVE: config.source.archives,
                    Group.TRANSFORM: config.transformations,
                    Group.MODEL: self.model,
                    Group.LOSS: self.loss,
                },
                allowed={
                    Group.ARCHIVE: (Group.MANIFEST,),
                    Group.TRANSFORM: (Group.MANIFEST, Group.ARCHIVE),
                    Group.MODEL: (Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM),
                    Group.LOSS: (
                        Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM, Group.MODEL
                    ),
                },
                extra_reductions={
                    Group.BATCH: itertools.product(
                        (Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM),
                        (Group.MODEL, Group.LOSS),
                    ),
                },
            )

        return self


__all__ = ["TrainingRunConfig"]
