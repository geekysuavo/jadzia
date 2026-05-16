"""Data sources."""

# Standard
from bisect import bisect_right
from itertools import accumulate
from typing import Any, NamedTuple

# Dax
from jadzia.data.archive import DataArchive
from jadzia.data.sharedmemory import SharedDataFrame


class ArchiveMapping(NamedTuple):
    """
    Mapping from data frame source values to archive keys.

    Attributes:
        archive: Name of the archive (`archives` dictionary key).
        source_key: Data frame column holding keys into the archive.
        dest_key: Optional data frame column to place archive values into.
    """

    archive: str
    source_key: str
    dest_key: str | None = None


class DataSource:
    """Random-access data source combining shared data frames and archives."""

    def __init__(
        self,
        dfs: dict[str, SharedDataFrame],
        archives: dict[str, DataArchive],
        mappings: dict[str, list[ArchiveMapping]],
    ):
        """
        Initialize the data source.

        Args:
            dfs: Named shared memory data frames.
            archives: Named on-disk data archives.
            mappings: Mapping from data frames to associated archives.
        """
        self._dfs = dfs
        self._archives = archives
        self._mappings = mappings

        self._keys = tuple(sorted(dfs))
        self._sizes = tuple(len(dfs[key]) for key in self._keys)

        self._offsets = tuple(accumulate(self._sizes))
        self._len = self._offsets[-1]

    def __len__(self) -> int:
        """Total number of items available in the data source."""
        return self._len

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Read an item from the data source.

        Args:
            index: Index of the item.

        Returns:
            Retrieved item.
        """
        if index < 0:
            index = index + self._len

        if index < 0 or index >= self._len:
            raise IndexError(f"index {index} out of range [0,{self._len})")

        key_index = bisect_right(self._offsets, index)
        df_key = self._keys[key_index]

        if key_index == 0:
            df_index = index
        else:
            df_index = index - self._offsets[key_index - 1]

        row = self._dfs[df_key][df_index]
        for mapping in self._mappings[df_key]:
            archive_key = row[mapping.source_key]
            archive_value = self._archives[mapping.archive][archive_key]
            row[mapping.dest_key or mapping.source_key] = archive_value

        return row


__all__ = ["ArchiveMapping", "DataSource"]
