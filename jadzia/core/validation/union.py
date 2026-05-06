"""Utility functions for constructing tagged unions."""

# Standard
from typing import Any

# Third party
from pydantic import (
    BaseModel,
    Discriminator,
    SerializerFunctionWrapHandler,
    WrapSerializer,
)


def _discriminate_tagged_union(value: Any) -> str | None:
    """
    Extracts a discriminator value from configuration data.

    Args:
        value: Dictionary of configuration data.

    Returns:
        Value of the `type` key in the input dictionary or `None` if the input
        is not a dictionary or does not contain that key.

    Important:
        This function _pops_ the `type` key out of `value`
        whenever `value` is a dictionary.
    """
    if isinstance(value, dict):
        return value.pop("type")
    elif isinstance(value, BaseModel):
        return type(value).__name__
    else:
        return None


def _serialize_tagged_union_obj(
    value: Any,
    handler: SerializerFunctionWrapHandler,
) -> dict[str, Any]:
    """
    Re-inserts a discriminator value into serialized data.

    Args:
        value: Object to serialize.
        handler: Serialization handler function.

    Returns:
        Result of calling `handler` and adding a `type` key.
    """
    data = handler(value)
    data["type"] = type(value).__name__
    return data


class UnionDiscriminator(Discriminator):
    """Tagged union discriminator."""

    def __init__(self):
        """Initialize the discriminator."""
        super().__init__(discriminator=_discriminate_tagged_union)


class UnionSerializer(WrapSerializer):
    """Tagged union wrap serializer."""

    def __init__(self):
        """Initialize the wrap serializer."""
        super().__init__(func=_serialize_tagged_union_obj)


__all__ = ["UnionDiscriminator", "UnionSerializer"]
