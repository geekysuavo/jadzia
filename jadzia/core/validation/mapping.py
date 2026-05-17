"""Validated mapping classes."""

# Standard
from typing import Iterable, Iterator, TypeVar

# Third party
from pydantic import ConfigDict, RootModel

# Dax
from jadzia.core.validation.base import Config

ConfigT = TypeVar("ConfigT", bound=Config)


class MappingConfig[ConfigT](RootModel[dict[str, ConfigT]]):
    """Mapping that holds configuration classes as values."""

    model_config = ConfigDict(frozen=True)

    root: dict[str, ConfigT]

    def __len__(self) -> int:
        """
        Size of the mapping.

        Returns:
            Number of mapping entries.
        """
        return len(self.root)

    def __iter__(self) -> Iterator[str]:  # ty: ignore[invalid-method-override]
        """
        Iterate over mapping keys.

        Returns:
            Iterator over mapping keys.
        """
        return iter(self.root)

    def __contains__(self, key: str) -> bool:
        """
        Whether the mapping contains a given key.

        Args:
            key: Key to check.

        Returns:
            True if the mapping contains `key`.
        """
        return key in self.root

    def __getitem__(self, key: str) -> ConfigT:
        """
        Get a mapping configuration entry.

        Args:
            key: Key to access.

        Returns:
            Configuration entry associated with `key`.

        Raises:
            KeyError: If `key` is not contained in the mapping.
        """
        return self.root[key]

    def keys(self) -> Iterable[str]:
        """Iterate over mapping keys."""
        return self.root.keys()

    def values(self) -> Iterable[ConfigT]:
        """Iterate over mapping values."""
        return self.root.values()

    def items(self) -> Iterable[tuple[str, ConfigT]]:
        """Iterate over mapping key-value pairs."""
        return self.root.items()


__all__ = ["MappingConfig"]
