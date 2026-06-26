"""Data configuration."""

# Standard
from collections.abc import Iterable

# Third party
import grain

# Dax
from jadzia.config.data.option import ReaderConfig, ShardingConfig, WorkerConfig
from jadzia.config.data.source import SourceMappingConfig
from jadzia.config.data.transform import BatchConfig, CompositeTransform
from jadzia.config.sampler import Sampler
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import MappingConfig


class DataLoaderConfig(Config):
    """
    Data loader configuration.

    Attributes:
        source: Data source configuration.
        transformations: Transform configurations.
        batch: Batching configuration.
        sampler: Sampler configuration.
        sharding: Sharding configuration.
        workers: Worker configuration.
        readers: Reader configuration.
    """

    source: SourceMappingConfig
    transformations: CompositeTransform
    batch: BatchConfig
    sampler: Sampler

    sharding: ShardingConfig = ShardingConfig()
    workers: WorkerConfig = WorkerConfig()
    readers: ReaderConfig = ReaderConfig()

    def build_instance(
        self,
        source_outputs: Iterable[Identifier] | None = None,
        transform_outputs: Iterable[Identifier] | None = None,
    ) -> grain.DataLoader:
        """Build an instance based on this configuration."""
        dfs = self.source.build_instance(outputs=source_outputs)
        (sampler, dfs) = self.sampler.build_instance(dfs)
        source = self.source.build_instance(dfs, outputs=source_outputs)
        del dfs

        batch_outputs = set().union(source_outputs or (), transform_outputs or ())
        operations = (
            *self.transformations.build_instance(transform_outputs),
            *self.batch.build_instance(batch_outputs or None),
        )
        return grain.DataLoader(
            data_source=source,
            sampler=sampler,
            operations=operations,
            worker_count=self.workers.count,
            worker_buffer_size=self.workers.buffer_size,
            shard_options=self.sharding.build_instance(),
            read_options=self.readers.build_instance(),
        )

    def build_subgraph(self) -> Graph[Identifier]:
        """Build a dependency graph based on this configuration."""
        return (
            self.source.build_subgraph()
            | self.batch.build_subgraph()
            | self.transformations.build_subgraph()
        )


class DataConfig(MappingConfig[DataLoaderConfig]):
    """Mapping of data loader configurations."""


__all__ = ["DataConfig", "DataLoaderConfig"]
