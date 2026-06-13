"""Data configuration."""

# Standard
from collections.abc import Iterable

# Third party
import grain
from pydantic import PositiveInt

# Dax
from jadzia.config.transform import Transform
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.utility import sort_gathers
from jadzia.core.validation.base import Config
from jadzia.core.validation.mapping import NodeMappingConfig
from jadzia.data.transform.base import Transformations


class BatchConfig(Config):
    """
    Batching configuration.

    Attributes:
        size: Batch size, number of elements to batch together.
        drop: Whether to drop final batches when they have a smaller size.
    """

    size: PositiveInt
    drop: bool = False

    def build_instance(
        self,
        outputs: Iterable[Identifier] | None = None,
    ) -> grain.transforms.Batch:
        """
        Build an instance from this configuration.

        Args:
            outputs: Iterable of transformation outputs to include.

        Returns:
            Newly built batch transformation.
        """
        # TODO -- pass outputs into a custom batch_fn
        return grain.transforms.Batch(batch_size=self.size, drop_remainder=self.drop)


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
