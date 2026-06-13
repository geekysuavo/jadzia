"""Data loader options configuration."""

# Third party
import grain
import jax
from pydantic import NonNegativeInt, PositiveInt

# Dax
from jadzia.core.validation.base import Config


class ShardingConfig(Config):
    """
    Sharding configuration.

    See [grain.sharding.ShardOptions][] for more information.

    Attributes:
        drop: When true, create even splits and drop remainder examples when sharding
            the dataset, ensuring the same number of records per shard. When false,
            distribute the remainder over the first shards.
    """

    drop: bool = False

    def build_instance(self) -> grain.sharding.ShardOptions:
        """Build an instance from this configuration."""
        if jax.process_count() > 1:
            return grain.sharding.ShardByJaxProcess(drop_remainder=self.drop)
        else:
            return grain.sharding.NoSharding()


class ReaderConfig(Config):
    """
    Data loader reader configuration.

    See [grain.ReadOptions][] for more information.

    Attributes:
        num_threads: Number of reader threads.
        buffer_size: Prefetch buffer size per reader process.
    """

    num_threads: NonNegativeInt = 0
    buffer_size: PositiveInt = 500

    def build_instance(self) -> grain.ReadOptions:
        """Build an instance from this configuration."""
        return grain.ReadOptions(
            num_threads=self.num_threads,
            prefetch_buffer_size=self.buffer_size,
        )


class WorkerConfig(Config):
    """
    Data loader worker configuration.

    Attributes:
        count: Number of child processes launched to parallelize transformations.
            Set to zero to use the main process. Set to `None` to auto-detect.
        buffer_size: Number of output batches to produce in advance per worker.
    """

    count: NonNegativeInt | None = 0
    buffer_size: PositiveInt = 1


__all__ = ["ReaderConfig", "ShardingConfig", "WorkerConfig"]
