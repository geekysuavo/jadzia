"""Data source configuration."""

# Standard
from collections.abc import Iterable, Sequence
from typing import ClassVar, overload

# Third party
import jax
import polars as pl
from pydantic import FilePath

# Dax
from jadzia.config.data.archive import ArchiveConfig
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, Reserved
from jadzia.core.hashing import hash_df
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import MappingConfig
from jadzia.data.sharedmemory import SharedDataFrame
from jadzia.data.source import ArchiveMapping, DataSource


class SourceConfig(Config):
    """
    Configuration for a single manifest and its associated archives.

    This class maintains a cache of shared data frames, keyed by checksum.
    The cache may be cleared by calling `clear_cache()`.

    Attributes:
        manifest: File holding manifest data in `parquet` format.
        archives: Dictionary of archive configurations, keyed by their
            destination column in the data source.
    """

    _cache: ClassVar[dict[str, SharedDataFrame]] = {}

    manifest: FilePath
    archives: dict[str, ArchiveConfig]

    def build_subgraph(self, name: str) -> Graph[Identifier]:
        """
        Build a dependency graph based on this configuration.

        Args:
            name: Name of the data source.

        Returns:
            Newly built dependency graph.
        """
        graph = Graph[Identifier]()
        root = Identifier.gather(name)
        graph.update(root)
        for node in (
            Reserved.ID(DataSource.MANIFEST_KEY),
            *map(Reserved.SOURCE, pl.read_parquet_schema(self.manifest)),
        ):
            graph[node] = (root,)

        for dest_key, config in self.archives.items():
            dest_node = Reserved.SOURCE(dest_key)
            if config.from_key:
                arch = Identifier.gather(_archive_name(name, dest_key))
                source_node = Reserved.SOURCE(config.from_key)
                graph.update(arch)
                graph[source_node] = (root,)
                graph[arch] = (source_node,)
                graph[dest_node] = (arch,)
            else:
                graph[dest_node] = (root,)

        return graph

    def read(self) -> pl.DataFrame:
        """Read the `manifest` file into a data frame."""
        return pl.read_parquet(self.manifest)

    def share(self, df: pl.DataFrame) -> SharedDataFrame:
        """Copy a data frame into shared memory and return the shared copy."""
        cache_key = hash_df(df)
        if cache_key in SourceConfig._cache:
            return SourceConfig._cache[cache_key]

        shared_df = SharedDataFrame(df, create=jax.process_index() == 0)
        SourceConfig._cache[cache_key] = shared_df
        return shared_df

    @staticmethod
    def clear_cache():
        """Clear the cache of shared data frames."""
        SourceConfig._cache.clear()


class SourceMappingConfig(MappingConfig[SourceConfig]):
    """Configuration for [jadzia.data.source.DataSource][]."""

    @property
    def archives(self) -> set[str]:
        """The graph node names of all data archives."""
        return {
            _archive_name(name, key)
            for name, config in self.items()
            for key in config.archives
        }

    @overload
    def build_instance(
        self,
        dfs: None = None,
        *,
        outputs: Iterable[Identifier] | None = None,
    ) -> Sequence[pl.DataFrame]: ...

    @overload
    def build_instance(
        self,
        dfs: Sequence[pl.DataFrame],
        *,
        outputs: Iterable[Identifier] | None = None,
    ) -> DataSource: ...

    def build_instance(
        self,
        dfs: Sequence[pl.DataFrame] | None = None,
        *,
        outputs: Iterable[Identifier] | None = None,
    ) -> Sequence[pl.DataFrame] | DataSource:
        """
        Build an instance from this configuration.

        This method is intended to be called twice: once without arguments,
        and again with an argument. The first call returns a list of data frames
        for optional processing. The second call accepts the processed data frames
        and returns a built data source.

        Processing may manipulate the data frames jointly, and may add, remove, or
        reorder rows and/or columns. However, it must not modify the order of the
        data frames.

        Args:
            dfs: Processed data frames to pass to the second call.
            outputs: Optional set of output names to include in the data source.

        Returns:
            Data frames or a newly built data source, depending on arguments.
        """
        names = tuple(sorted(self))
        if dfs is None:
            return [self[name].read() for name in names]

        dfs = [
            df.rename({column: str(Reserved.SOURCE(column)) for column in df.columns})
            for df in dfs
        ]

        if outputs:
            output_columns = set(map(str, outputs))
            dfs = [df.drop(set(df.columns) - output_columns) for df in dfs]
        else:
            output_columns = set().union(*(df.columns for df in dfs))
            output_columns.add(str(Reserved.ID(DataSource.MANIFEST_KEY)))

        shared_dfs = [self[name].share(df) for name, df in zip(names, dfs)]
        del dfs

        archives = {}
        mappings = {}
        for name in names:
            for dest_key, config in self[name].archives.items():
                archive_name = _archive_name(name, dest_key)
                source_key = config.from_key or dest_key
                mapping = ArchiveMapping(
                    archive_name,
                    source_key=str(Reserved.SOURCE(source_key)),
                    dest_key=str(Reserved.SOURCE(dest_key)),
                )

                archives[archive_name] = config.build_instance()
                mappings.setdefault(name, []).append(mapping)

        return DataSource(
            dfs=dict(zip(names, shared_dfs)),
            archives=archives,
            mappings=mappings,
            insert_id=True,
            outputs=output_columns,
        )

    def build_subgraph(self) -> Graph[Identifier]:
        """Build a dependency graph based on this configuration."""
        graphs = (config.build_subgraph(name) for name, config in self.items())
        return Graph.union(*graphs)


def _archive_name(data_source_name: str, archive_dest_key: str) -> str:
    """Return the name of an archive associated with a data source."""
    return f"{data_source_name}[{archive_dest_key}]"


__all__ = ["SourceConfig", "SourceMappingConfig"]
