"""Extensions for serializing [numpy][] arrays."""

# Standard
from typing import Any

# Third party
import msgpack
import numpy as np


def _encode_ndarray(obj: Any) -> Any:
    """Encode a [numpy.ndarray][] into a serializable data structure."""
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.name,
            "shape": obj.shape,
        }
    else:
        return obj


def _decode_ndarray(obj: dict[str, Any]) -> np.ndarray | dict[str, Any]:
    """Decode a deserialized data structure into a [numpy.ndarray][]."""
    if "__ndarray__" in obj:
        return np.frombuffer(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
    else:
        return obj


def dumps(obj: Any) -> bytes:
    """
    Serialize an object into [bytes][].

    Args:
        obj: Object to serialize.

    Returns:
        Serialized data.
    """
    return msgpack.packb(obj, default=_encode_ndarray)


def loads(data: bytes) -> Any:
    """
    Deserialize [bytes][] into an object.

    Args:
        data: Data to deserialize.

    Returns:
        Deserialized object.
    """
    return msgpack.unpackb(data, object_hook=_decode_ndarray)


__all__ = ["dumps", "loads"]
