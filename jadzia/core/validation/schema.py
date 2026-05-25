"""Schema versioning."""

# Standard
import importlib.metadata
from typing import Annotated, NamedTuple, Self

# Third party
from pydantic import BeforeValidator, PlainSerializer


class SchemaVersion(NamedTuple):
    """
    Configuration schema version tuple.

    Attributes:
        major: Major version.
        minor: Minor version.
    """

    major: int
    minor: int

    def __str__(self) -> str:
        """Return a string representation of this instance."""
        return f"{self.major}.{self.minor}"

    @classmethod
    def from_metadata(cls) -> Self:
        """Get the schema version expected by the current package."""
        value = importlib.metadata.version("jadzia")
        (major, minor, *_) = value.split(".")
        (major, minor) = map(int, (major, minor))
        return cls(major, minor)

    @classmethod
    def from_str(cls, value: str, *, check: bool = True) -> Self:
        """
        Convert a string into a schema version.

        Args:
            value: String value to convert.
            check: When true, check that the specified version matches the package.

        Returns:
            Parsed schema version.

        Raises:
            TypeError: If the value is not a string.
            ValueError: If the value is not a proper schema version string,
                or if the value does not match the current package version.
        """
        if not isinstance(value, str):
            raise TypeError(f"expected str, got {type(value).__name__}")

        parts = value.split(".")
        if len(parts) != 2:
            raise ValueError(f"expected 'major.minor' format, got {value!r}")

        (major, minor) = map(int, parts)
        version = cls(major, minor)

        required = cls.from_metadata() if check else None
        if required is not None and version != required:
            raise ValueError(f"expected schema version {required}, got {version}")

        return version


type SchemaVersionStr = Annotated[
    SchemaVersion,
    BeforeValidator(SchemaVersion.from_str),
    PlainSerializer(SchemaVersion.__str__),
]
"""Schema version string field annotation."""


__all__ = ["SchemaVersion", "SchemaVersionStr"]
