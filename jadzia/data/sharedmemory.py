"""Classes for sharing data frame rows across processes."""

# Standard
import contextlib
import ctypes
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Self

# Third party
import polars as pl

# Dax
from jadzia.core.hashing import hash_df
from jadzia.data.serialization import dumps, loads

_INDEX_SIZE: int = ctypes.sizeof(ctypes.c_size_t)
"""Number of bytes used to store each index offset."""

_WAIT_INTERVAL: int = 5
"""Number of seconds to wait between status checks when attaching shared memory."""


class _SharedDataFrameIterator:
    """Iterator over shared memory data frames."""

    def __init__(self, source: "SharedDataFrame"):
        """
        Initialize the iterator.

        Args:
            source: Shared data frame to iterate over.
        """
        self._source = source
        self._index = 0

    def __iter__(self) -> Self:
        """Reset and return this iterator."""
        self._index = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Return the next item from the shared data frame."""
        if self._index >= len(self._source):
            raise StopIteration

        item = self._source[self._index]
        self._index += 1
        return item


def _item_size(item: tuple) -> int:
    """
    Determine the size of a single item once pickled.

    Args:
        item: Named tuple from the initial data frame.

    Returns:
        Number of bytes needed to store `item`.
    """
    return len(dumps(item))


class SharedDataFrame:
    """Shared memory block holding a list of data frame row dictionaries."""

    def __init__(
        self,
        df: pl.DataFrame,
        *,
        name: str | None = None,
        create: bool = False,
    ):
        """
        Initialize the shared memory data frame.

        Args:
            df: Input data frame.
            name: Unique name of the shared memory block. If unspecified,
                the data frame will be hashed to determine a name.
            create: When true, the shared memory block will be created.
                When false, an existing shared memory block of the same
                `name` will be attached to.
        """
        self._len = len(df)
        self._owner = create
        self._keys = tuple(df.columns)

        if name is None:
            name = hash_df(df, length=16)

        if not create:
            self._attach(name)
            return

        num_item_bytes = sum(map(_item_size, df.iter_rows()))
        num_offset_bytes = _INDEX_SIZE * (self._len + 2)
        num_bytes = num_item_bytes + num_offset_bytes

        self._shm = SharedMemory(name, create=True, size=num_bytes)
        assert self._shm.buf is not None

        begin = num_offset_bytes
        for index, item in enumerate(df.iter_rows()):
            item_data = dumps(item)
            item_size = len(item_data)
            end = begin + item_size

            self._set_ints(index, begin, end)
            self._set_bytes(begin, end, item_data)

            begin = end

        self._shm.buf[0] = 0xFF

    def _attach(self, name: str):
        """
        Attach to an existing shared memory block.

        Args:
            name: Name of the shared memory block.

        Note:
            This method will wait indefinitely for the block to exist
            and be marked as ready by the creating shared data frame.
        """
        attached = False
        while not attached:
            try:
                self._shm = SharedMemory(name)
                attached = True
            except FileNotFoundError:
                time.sleep(_WAIT_INTERVAL)

        while self._shm.buf and not self._shm.buf[0]:
            time.sleep(_WAIT_INTERVAL)

    def _get_bytes(self, begin: int, end: int) -> bytes:
        """Read a chunk of data from the shared memory block."""
        assert self._shm.buf is not None
        return bytes(self._shm.buf[begin:end])

    def _set_bytes(self, begin: int, end: int, data: bytes):
        """Write a chunk of data to the shared memory block."""
        assert self._shm.buf is not None
        self._shm.buf[begin:end] = data

    def _get_ints(self, index: int) -> tuple[int, int]:
        """Read a pair of integers from the shared memory block."""
        begin = _INDEX_SIZE * (index + 1)
        end = begin + _INDEX_SIZE * 2
        data = self._get_bytes(begin, end)
        first = int.from_bytes(data[:_INDEX_SIZE])
        second = int.from_bytes(data[_INDEX_SIZE:])
        return (first, second)

    def _set_ints(self, index: int, first: int, second: int):
        """Write a pair of integers to the shared memory block."""
        begin = _INDEX_SIZE * (index + 1)
        end = begin + _INDEX_SIZE * 2
        first_data = first.to_bytes(length=_INDEX_SIZE)
        second_data = second.to_bytes(length=_INDEX_SIZE)
        self._set_bytes(begin, end, first_data + second_data)

    def __len__(self) -> int:
        """Number of items stored in shared memory."""
        return self._len

    def __iter__(self) -> _SharedDataFrameIterator:
        """Return an iterator over the stored items."""
        return _SharedDataFrameIterator(self)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Read a stored item from shared memory.

        Args:
            index: Index of the item.

        Returns:
            Retrieved item.
        """
        if index < 0:
            index = index + self._len

        if index < 0 or index >= self._len:
            raise IndexError(f"index {index} out of range [0,{self._len})")

        (begin, end) = self._get_ints(index)
        values = loads(self._get_bytes(begin, end))
        return dict(zip(self._keys, values))

    def __del__(self):
        """Unlink the shared memory block upon deletion."""
        with contextlib.suppress(Exception):
            self._shm.close()
            if self._owner:
                self._shm.unlink()


__all__ = ["SharedDataFrame"]
