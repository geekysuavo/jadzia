"""Reserved names for dependency management."""

# Standard
from enum import StrEnum

# Dax
from jadzia.core.dependency.identifier import Identifier


class Reserved(StrEnum):
    """
    Enumeration of reserved node names.

    - `id`: Identifying data inserted by the data source.
    - `source`: Value provided by the data source manifest or archives.
    """

    ID = "id"
    SOURCE = "source"

    def __call__(self, output: str) -> Identifier:
        """
        Build an identifier by combining this node name with an output name.

        Args:
            output: Output name.

        Returns:
            Identifier with this node name and the specified output name.
        """
        return Identifier(self.value, output)


__all__ = ["Reserved"]
