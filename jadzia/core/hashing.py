"""Hashing functionality."""

# Standard
import hashlib

# Third party
import polars as pl


def hash_df(df: pl.DataFrame, *, length: int | None = None) -> str:
    """
    Build a checksum from a data frame.

    Args:
        df: Data frame to hash.
        length: Optional string length limit.

    Returns:
        Hexadecimal checksum of the data frame.
    """
    hasher = hashlib.sha256()

    for column, dtype in df.schema.items():
        hasher.update(column.encode())
        hasher.update(str(dtype).encode())

    for row_hash in df.hash_rows():
        hasher.update(row_hash.to_bytes(64))

    if length is None:
        return hasher.hexdigest()
    else:
        return hasher.hexdigest()[:length]


__all__ = ["hash_df"]
