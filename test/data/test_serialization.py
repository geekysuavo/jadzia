"""Tests of serialization."""

# Standard
from types import SimpleNamespace

# Third party
import numpy as np
import pytest

# Dax
from jadzia.data.serialization import dumps, loads


@pytest.mark.parametrize(
    "shape",
    [(4,), (2, 3), (5, 3, 2)],
    ids=["4", "2x3", "5x3x2"],
)
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
        "complex128",
    ],
)
def test_ndarray_serialization(dtype, shape):
    """Numpy arrays are serialized and deserialized correctly."""
    rng = np.random.default_rng()
    if "int" in dtype:
        array = rng.integers(0, 100, size=shape, dtype=dtype)
    else:
        array = rng.normal(size=shape).astype(dtype)

    if "complex" in dtype:
        array *= (1 + 1j)

    obj = {"array": array, "i": 42, "z": 1.125, "hello": ["world"]}

    data = dumps(obj)
    assert isinstance(data, bytes)

    obj_copy = loads(data)
    assert obj_copy is not obj

    assert np.array_equal(obj_copy["array"], obj["array"])

    obj_copy.pop("array")
    obj.pop("array")
    assert obj_copy == obj


def test_unknown_type():
    """Unknown types will pass through the encoder and raise an exception."""
    obj = SimpleNamespace()
    with pytest.raises(TypeError, match=r"can not serialize 'types.SimpleNamespace'"):
        dumps(obj)
