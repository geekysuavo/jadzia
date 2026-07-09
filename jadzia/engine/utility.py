"""Engine utility functions."""

# Standard
import functools
from collections.abc import Callable

# Third party
import jax

type SendToDeviceFunc = Callable[[dict[str, jax.Array]], dict[str, jax.Array]]
"""Function sending a dictionary of arrays to the appropriate device(s)."""


def make_send_to_device(sharding: jax.sharding.Sharding) -> SendToDeviceFunc:
    """
    Return a function that sends arrays to devices.

    Args:
        sharding: Sharding of the global arrays.

    Returns:
        Function that sends array data to the appropriate device(s).
    """
    return functools.partial(jax.make_array_from_process_local_data, sharding)


__all__ = ["make_send_to_device"]
