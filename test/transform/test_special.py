"""Tests of special transformations."""

# Third party
import numpy as np

# Dax
from jadzia.data.transform.special import ConvertNone, SelectKeys


def test_convert_none():
    """Nones may be converted to NaN."""
    transform = ConvertNone()
    element = {"hello": 42, "world": None}
    assert transform.map(element) == {"hello": 42, "world": np.nan}


def test_select_keys():
    """Keys may be selected from dictionaries."""
    transform = SelectKeys(("foo", "bar"))
    element = {"foo": 1, "bar": 2.5, "baz": "hello"}
    assert transform.map(element) == {"foo": 1, "bar": 2.5}
