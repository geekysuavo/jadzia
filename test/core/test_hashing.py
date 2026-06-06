"""Tests of hashing functionality."""

# Third party
import polars as pl
import pytest

# Dax
from jadzia.core.hashing import hash_df


@pytest.mark.parametrize("length", [None, 32, 16])
def test_hash_df(length):
    """Data frames may be hashed into strings."""
    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.5, 6.7, 8.9], "z": ["z"] * 3})
    value = hash_df(df, length=length)
    assert isinstance(value, str)
    if length is not None:
        assert len(value) == length
    else:
        assert len(value) == 64
