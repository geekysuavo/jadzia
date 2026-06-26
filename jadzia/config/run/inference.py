"""Inference run configuration."""

# Standard
import itertools
from typing import Self

# Third party
import jax.random as jr
from pydantic import PrivateAttr, model_validator

# Dax
from jadzia.config.engine import InferenceEngineConfig
from jadzia.config.run.base import Group, RunConfig
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, Reserved
from jadzia.core.validation.dataflow import validate_data_flow
from jadzia.engine.inference import InferArgs


class InferenceRunConfig(RunConfig):
    """
    Configuration of a `jadzia` inference run.

    Attributes:
        version: Configuration schema version.
        data: Dataset configurations.
        model: Model configuration.
        engine: Inference configuration.
    """

    engine: InferenceEngineConfig

    _separators: dict[str, set[Identifier]] = PrivateAttr({})

    def build_instance(self) -> InferArgs:
        """Build an instance from this configuration."""
        manifest_outputs = self._separators[Group.MANIFEST]
        archive_outputs = self._separators[Group.ARCHIVE]
        batch_outputs = self._separators[Group.BATCH]
        model_outputs = self._separators[Group.MODEL]

        dataset_name = self.engine.inference.dataset
        dataloader = self.data[dataset_name].build_instance(
            source_outputs=manifest_outputs | archive_outputs,
            batch_outputs=batch_outputs,
        )

        (model, model_state) = self.model.build_instance(
            jr.key(self.context.seed) if self.model.requires_key else None,
            outputs=model_outputs,
            with_state=True,
        )

        if self.engine.checkpoints is not None:
            checkpointer = self.engine.checkpoints.build_instance()
        else:
            checkpointer = None

        return InferArgs(
            model=model,
            model_state=model_state,
            dataloader=dataloader,
            checkpointer=checkpointer,
            capture=tuple(map(str, dict.fromkeys(self.engine.inference.capture))),
            checkpoint_dir=self.engine.inference.checkpoint_dir,
            output_file=self.engine.inference.output_file,
        )

    def build_subgraph(self) -> tuple[Graph[Identifier], Identifier]:
        """
        Build a dependency graph based on this configuration.

        Returns:
            Newly built dependency graph.
            Terminal node of the dependency graph.
        """
        dataset_name = self.engine.inference.dataset
        data_graph = self.data[dataset_name].build_subgraph()
        model_graph = self.model.build_subgraph()
        (eval_graph, terminal) = self.engine.inference.build_subgraph()
        return (Graph.union(data_graph, model_graph, eval_graph), terminal)

    @model_validator(mode="after")
    def _validate_data_flow(self) -> Self:
        """Validate the data flow graph."""
        required = {self.engine.inference.dataset}
        if missing := required - set(self.data):
            raise ValueError(f"datasets are missing: {', '.join(missing)}")
        elif extra := set(self.data) - required:
            raise ValueError(f"datasets are unused: {', '.join(extra)}")

        dataset_name = self.engine.inference.dataset
        config = self.data[dataset_name]

        self._separators = validate_data_flow(
            *self.build_subgraph(),
            assignment={
                Group.MANIFEST: config.source,
                Group.ARCHIVE: config.source.archives,
                Group.TRANSFORM: config.transformations,
                Group.MODEL: self.model,
                Group.EVAL: (Reserved.EVAL,),
            },
            allowed={
                Group.ARCHIVE: (Group.MANIFEST,),
                Group.TRANSFORM: (Group.MANIFEST, Group.ARCHIVE),
                Group.MODEL: (Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM),
                Group.EVAL: (
                    Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM, Group.MODEL
                ),
            },
            extra_reductions={
                Group.BATCH: itertools.product(
                    (Group.MANIFEST, Group.ARCHIVE, Group.TRANSFORM),
                    (Group.MODEL, Group.EVAL),
                ),
            },
        )
        return self


__all__ = ["InferenceRunConfig"]
