"""Tests of transformation base classes."""

# Standard
from dataclasses import dataclass

# Third party
import grain.transforms as gt
import numpy as np
import pytest

# Dax
from jadzia.data.transform.base import (
    Filter,
    Map,
    MapWithIndex,
    NumpyRandomMap,
    RandomMap,
)


@dataclass(frozen=True)
class InsertInt(Map):
    value: int

    def __call__(self) -> int:
        return self.value


@dataclass(frozen=True)
class Swap(Map):
    def __call__(self, a: int, b: int) -> tuple[int, int]:
        return (b, a)


@dataclass(frozen=True)
class InsertIndex(MapWithIndex):
    def __call__(self, index: int) -> int:
        return index


@dataclass(frozen=True)
class InsertRandomScalar(RandomMap):
    def __call__(self, upper: int, rng: np.random.Generator) -> int:
        return rng.integers(upper).item()


@dataclass(frozen=True)
class InsertRandomArray(NumpyRandomMap):
    def __call__(self, size: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
        return rng.normal(size=size)


@dataclass(frozen=True)
class IsEven(Filter):
    def __call__(self, value: int) -> bool:
        return value % 2 == 0


@pytest.mark.parametrize("value", [5, 42])
@pytest.mark.parametrize("key", ["foo", "bar"])
def test_map(key, value):
    """Map transformations produce the expected results."""
    transform = InsertInt(input_names=(), output_names=(key,), value=value)
    assert isinstance(transform, gt.Map)

    element_in = {"pre": "v"}
    element_out = transform.map(element_in)

    assert element_out is not element_in
    assert element_out == {key: value, "pre": "v"}


@pytest.mark.parametrize(("value_a", "value_b"), [(2, 3), (5, 7)])
@pytest.mark.parametrize(("key_a", "key_b"), [("foo", "bar"), ("Bar", "Foo")])
def test_swap(key_a, key_b, value_a, value_b):
    """Map transformations may return or modify multiple values."""
    transform = Swap(input_names=(key_a, key_b), output_names=(key_a, key_b))
    assert isinstance(transform, gt.Map)

    element_in = {key_a: value_a, key_b: value_b}
    element_out = transform.map(element_in)

    assert element_out is not element_in
    assert element_out == {key_a: value_b, key_b: value_a}


@pytest.mark.parametrize("index", [0, 3])
@pytest.mark.parametrize("key", ["foo", "bar"])
def test_map_with_index(key, index):
    """Indexed map transformations produce the expected results."""
    transform = InsertIndex(input_names=(), output_names=(key,))
    assert isinstance(transform, gt.MapWithIndex)

    element_in = {"pre": "V"}
    element_out = transform.map_with_index(index, element_in)

    assert element_out is not element_in
    assert element_out == {key: index, "pre": "V"}


@pytest.mark.parametrize("upper", [23, 57])
@pytest.mark.parametrize(("key_in", "key_out"), [("foo", "bar"), ("Bar", "Foo")])
def test_random_map(key_in, key_out, upper):
    """Random map transformations produce the expected results."""
    transform = InsertRandomScalar(input_names=(key_in,), output_names=(key_out,))
    assert isinstance(transform, gt.RandomMap)

    rng = np.random.default_rng()
    element_in = {key_in: upper, "some": "else"}
    element_out = transform.random_map(element_in, rng)

    assert element_out is not element_in
    assert set(element_out) == {key_in, key_out, "some"}
    assert element_out[key_in] == upper
    assert element_out["some"] == "else"
    assert isinstance(element_out[key_out], int)
    assert element_out[key_out] in range(upper)


@pytest.mark.parametrize("size", [(4,), (2, 3)], ids=["4", "2x3"])
@pytest.mark.parametrize(("key_in", "key_out"), [("foo", "bar"), ("Bar", "Foo")])
def test_np_random_map(key_in, key_out, size):
    """Random numpy map transformations produce the expected results."""
    transform = InsertRandomArray(input_names=(key_in,), output_names=(key_out,))
    assert isinstance(transform, gt.TfRandomMap)

    rng = np.random.default_rng()
    element_in = {key_in: size, "some": "Else"}
    element_out = transform.np_random_map(element_in, rng)

    assert element_out is not element_in
    assert set(element_out) == {key_in, key_out, "some"}
    assert element_out[key_in] == size
    assert element_out["some"] == "Else"
    assert isinstance(element_out[key_out], np.ndarray)
    assert element_out[key_out].shape == size


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, True), (2, True), (1, False), (3, False)],
)
@pytest.mark.parametrize("key", ["Foo", "Bar"])
def test_filter(key, value, expected):
    """Filter transformations produce the expected results."""
    transform = IsEven(input_names=(key,), output_names=())
    assert isinstance(transform, gt.Filter)

    element_in = {key: value, "other": 3.25}
    returned = transform.filter(element_in)

    assert isinstance(returned, bool)
    assert returned == expected
