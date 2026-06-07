"""Validation base class."""

# Standard
from pathlib import Path
from typing import Any, Self

# Third party
import yaml
from pydantic import BaseModel, ConfigDict


class Config(BaseModel):
    """Base class for all validated configuration objects."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_yaml(self, file: Path):
        """
        Save a configuration object to a `yaml` file.

        Args:
            file: File to write configuration data into.
        """
        data = self.model_dump(mode="json", exclude_defaults=True, exclude_unset=True)
        with file.open("w") as fh:
            yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, file: Path, context: Any | None = None) -> Self:
        """
        Load a configuration object from a `yaml` file.

        Args:
            file: File to read configuration data from.
            context: Optional validation context.

        Returns:
            New validated configuration object.
        """
        with file.open() as fh:
            data = yaml.safe_load(fh)

        return cls.model_validate(data, context=context)


__all__ = ["Config"]
