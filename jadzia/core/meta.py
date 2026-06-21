"""Metaclass functionality."""

# Standard
from abc import ABCMeta
from dataclasses import dataclass
from typing import dataclass_transform


@dataclass_transform(frozen_default=True)
class AbstractDataclassMeta(ABCMeta):
    """
    Minimal metaclass providing [abc][] and [dataclasses][] semantics.

    Classes created from this metaclass will exhibit frozen dataclass behavior
    and support abstract methods, two behaviors that are generally useful.
    """

    def __new__(cls, name, bases, dct):
        """Customize class behavior."""
        new_cls = super().__new__(cls, name, bases, dct)
        return dataclass(new_cls, frozen=True)


__all__ = ["AbstractDataclassMeta"]
