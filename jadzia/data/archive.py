"""Data archives."""

# Standard
import contextlib
from pathlib import Path
from typing import Any, Iterator

# Third party
import lmdb

# Dax
from jadzia.data.serialization import dumps, loads

_MAP_SIZE: int = 100 * 1024**3
"""Maximum size a created database may grow to, in bytes."""


class DataArchive:
    """Key-value store of data in [lmdb](https://lmdb.readthedocs.io/) format."""

    def __init__(self, path: Path, read_only: bool = True):
        """
        Initialize the archive.

        Args:
            path: Archive directory. Will be created if `read_only` is false.
            read_only: When true, expect `path` to exist and only allow reading.
        """
        if read_only and not path.is_dir():
            raise FileNotFoundError(f"expected directory: {path}")

        self._read_only = read_only
        self._path = str(path)
        self._open_env()

    def __len__(self) -> int:
        """Number of items in the archive."""
        with self._env.begin() as txn:
            return txn.stat().get("entries", 0)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over stored keys."""
        return self.keys()

    def __contains__(self, key: str) -> bool:
        """Whether the archive contains a given key."""
        with self._env.begin() as txn:
            return txn.get(key.encode()) is not None

    def __getitem__(self, key: str) -> Any:
        """
        Get the value associated with a given key.

        Args:
            key: Key to access.

        Returns:
            Value associated with the key.

        Raises:
            KeyError: If `key` is not in the archive.
        """
        with self._env.begin(buffers=True) as txn:
            data = txn.get(key.encode())

        if data is not None:
            return loads(data)
        else:
            raise KeyError(key)

    def __setitem__(self, key: str, value: Any):
        """
        Set the value associated with a given key.

        Args:
            key: Key to modify.
            value: Value to store.

        Raises:
            RuntimeError: If the archive is read only.
        """
        if self._read_only:
            raise RuntimeError("archive is read-only")

        with self._env.begin(write=True) as txn:
            txn.put(key.encode(), dumps(value))

    def __delitem__(self, key: str):
        """
        Delete the value associated with a given key.

        Args:
            key: Key to access.

        Raises:
            KeyError: If `key` is not in the archive.
            RuntimeError: If the archive is read only.
        """
        if self._read_only:
            raise RuntimeError("archive is read-only")

        with self._env.begin(write=True) as txn:
            deleted = txn.delete(key.encode())

        if not deleted:
            raise KeyError(key)

    def __del__(self):
        """Close the `lmdb.Environment` upon deletion."""
        with contextlib.suppress(Exception):
            self._env.sync()
            self._env.close()

    def __getstate__(self) -> dict:
        """Return object state for serialization."""
        state = self.__dict__.copy()
        state["_env"] = None
        return state

    def __setstate__(self, state: dict):
        """Reload object state from deserialization."""
        self.__dict__.update(state)
        self._open_env()

    def _open_env(self):
        """Open the `lmdb.Environment` object."""
        self._env = lmdb.open(
            self._path,
            readonly=self._read_only,
            lock=not self._read_only,
            map_size=_MAP_SIZE,
        )

    def keys(self) -> Iterator[str]:
        """Return an iterator over archive keys."""
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key in cursor.iternext(values=False):
                assert isinstance(key, memoryview)
                yield key.tobytes().decode()

    def values(self) -> Iterator[Any]:
        """Return an iterator over archive values."""
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for data in cursor.iternext(keys=False):
                assert isinstance(data, memoryview)
                yield loads(data)

    def items(self) -> Iterator[tuple[str, Any]]:
        """Return an iterator over archive key-value pairs."""
        with self._env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, data in cursor:
                assert isinstance(key, memoryview)
                assert isinstance(data, memoryview)
                yield (key.tobytes().decode(), loads(data))

    def update(self, updates: dict[str, Any]):
        """
        Update multiple key-value pairs in the archive.

        Args:
            updates: Dictionary of data to add to the archive.

        Note:
            This updates all key-value pairs in a single transaction.
        """
        if self._read_only:
            raise RuntimeError("archive is read-only")

        with self._env.begin(write=True) as txn:
            for key, value in updates.items():
                txn.put(key.encode(), dumps(value))


__all__ = ["DataArchive"]
