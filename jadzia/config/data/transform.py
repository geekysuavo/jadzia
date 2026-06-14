"""Data configuration."""

# Standard
from collections.abc import Iterable

# Third party
import grain
from pydantic import PositiveInt

# Dax
from jadzia.config.transform import Transform
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, IdentifierStr, Reserved
from jadzia.core.dependency.utility import sort_gathers
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import NodeMappingConfig
from jadzia.data.batching import Batcher
from jadzia.data.transform.base import Transformations
from jadzia.data.transform.special import ConvertNone, SelectKeys


class BatchConfig(Config):
    """
    Batching configuration.

    Attributes:
        size: Batch size, number of elements to batch together.
        drop: Whether to drop final batches when they have a smaller size.
        concatenate: Names to batch by padded concatenation instead of stacking.
        convert_none: Whether to convert `None` values to `np.nan` before batching.
    """

    size: PositiveInt
    drop: bool = False
    concatenate: set[IdentifierStr] | None = None
    convert_none: bool = True

    def build_instance(
        self,
        outputs: Iterable[Identifier] | None = None,
    ) -> grain.transforms.Transformations:
        """
        Build an instance from this configuration.

        Args:
            outputs: Iterable of transformation outputs to include.

        Returns:
            Newly built batch transformations.
        """
        transforms = ()
        if self.convert_none:
            transforms = (*transforms, ConvertNone())

        if outputs is not None:
            keys = tuple(map(str, outputs))
            transforms = (*transforms, SelectKeys(keys))

        batch_fn = Batcher(concat=tuple(map(str, self.concatenate or ())))
        batch_transform = grain.transforms.Batch(
            batch_fn=batch_fn,
            batch_size=self.size,
            drop_remainder=self.drop,
        )
        return (*transforms, batch_transform)

    def build_subgraph(self) -> Graph[Identifier]:
        """Build a dependency graph based on this configuration."""
        graph = Graph[Identifier]()
        if self.concatenate is not None:
            gather = Identifier.gather(Reserved.BATCH)
            graph.update(gather, *self.concatenate)

        return graph


class CompositeTransform(NodeMappingConfig[Transform]):
    """Mapping of transformation configurations."""

    def build_instance(
        self,
        outputs: Iterable[Identifier] | None = None,
    ) -> Transformations:
        """
        Build an instance from this configuration.

        Args:
            outputs: Iterable of transformation outputs to include.

        Returns:
            Newly built sequence of transformations.
        """
        outputs = outputs or set(map(Identifier.gather, self))
        order = sort_gathers(self.build_subgraph(), subset=outputs)
        return tuple(self[name].build_instance(name) for name in order)


__all__ = ["BatchConfig", "CompositeTransform"]
