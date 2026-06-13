"""Data archive configuration."""

# Standard
from pathlib import Path
from typing import ClassVar

# Third party
from pydantic import DirectoryPath

# Dax
from jadzia.core.validation.base import Config
from jadzia.data.archive import DataArchive


class ArchiveConfig(Config):
    """
    Configuration for a single archive.

    This class maintains a cache of built data sources, keyed by archive path.
    The cache may be cleared by calling `clear_cache()`.

    Attributes:
        path: Directory holding archive data in `lmdb` format.
        from_key: Optional key to read from
    """

    _cache: ClassVar[dict[Path, DataArchive]] = {}

    path: DirectoryPath
    from_key: str | None = None

    def build_instance(self) -> DataArchive:
        """Build an instance from this configuration."""
        if self.path in ArchiveConfig._cache:
            return ArchiveConfig._cache[self.path]

        archive = DataArchive(self.path)
        ArchiveConfig._cache[self.path] = archive
        return archive

    @staticmethod
    def clear_cache():
        """Clear the cache of built data archives."""
        ArchiveConfig._cache.clear()


__all__ = ["ArchiveConfig"]
