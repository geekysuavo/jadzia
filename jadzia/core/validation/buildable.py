"""Validation base class for buildable objects."""

# Standard
from typing import Any, ClassVar, Self, TypeVar

# Third party
from pydantic import create_model

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.inspect import inspect_initializer

InstanceT = TypeVar("InstanceT")


class Exclude:
    """Sentinel class used to exclude fields when building instances."""


class BuildableConfig[InstanceT](Config):
    """Base class for all validated configurations of buildable objects."""

    instance_type: ClassVar[type]

    def model_dump_for_instance(
        self,
        *,
        pass_subclasses: tuple[type[Config], ...] = (),
    ) -> dict[str, Any]:
        """
        Dump configuration data, aliasing and/or excluding certain fields.

        This method is used to construct initializer arguments when building instances.
        ften, configuration classes contain aliased or additional fields that their
        associated buildable classes do not need, so this method makes handling those
        cases simpler.

        Args:
            pass_subclasses: Any field values that are instances of these classes
                will pass through serialization unchanged, i.e. they will not be
                converted into plain data.

        Returns:
            Dictionary that does not contain items for fields that were annotated
                by [jadzia.core.validation.buildable.Exclude][] in the class
                definition.
        """
        cls = type(self)
        exclude = {
            name
            for name, field_info in cls.model_fields.items()
            if Exclude in field_info.metadata
        }
        passthrough = {
            name
            for name in cls.model_fields
            if isinstance(getattr(self, name), pass_subclasses)
        }
        data = self.model_dump(by_alias=True, exclude=exclude | passthrough)
        data.update({name: getattr(self, name) for name in passthrough - exclude})
        return data

    def build_instance(self) -> InstanceT:
        """Build an instance based on this configuration."""
        cls = type(self)
        kwargs = self.model_dump_for_instance()
        return cls.instance_type(**kwargs)

    @classmethod
    def auto_hook(
        cls,
        instance_type: type[InstanceT],
        *,
        field_defs: dict[str, tuple[Any, Any]],
    ):
        """
        Class method used to customize the behavior of `auto`.

        Args:
            instance_type: Type to create a configuration class for.
            field_defs: Dictionary of field definitions obtained by inspection.
        """

    @classmethod
    def auto(
        cls,
        instance_type: type[InstanceT],
        *,
        location: str | None = None,
        **defaults: Any,
    ) -> type[Self]:
        """
        Dynamically create a new configuration class by inspecting a type.

        Args:
            instance_type: Type to create a configuration class for.
            location: Name of the module that the created class will belong to.
                When unset, a default will be chosen based on the call stack.
            defaults: Values used to set or override defaults of configuration fields.

        Returns:
            New subclass created by [pydantic.create_model][], holding configuration
                fields for all arguments to the initializer method of `instance_type`.

        Note:
            Subclasses may override the `auto_hook` method to augment the field
            definitions obtained by inspection.
        """
        field_defs: dict[str, tuple[Any, Any]] = {}
        field_defs["instance_type"] = (cls.find_instance_type(), instance_type)
        inspect_initializer(instance_type, field_defs=field_defs, **defaults)
        cls.auto_hook(instance_type, field_defs=field_defs)
        return create_model(  # ty: ignore[no-matching-overload]
            instance_type.__name__,
            __base__=cls,
            __doc__=instance_type.__doc__,
            __module__=location or __name__,
            **field_defs,
        )

    @classmethod
    def find_instance_type(cls) -> type:
        """
        Find the nearest `instance_type` annotation in the class hierarchy.

        The first parent class in the hierarchy is always traversed, under the
        assumption that secondary parents (e.g. `ABC`) are after the parent
        that holds the instance type field.
        """
        base_cls = cls
        while "instance_type" not in base_cls.__annotations__:
            if base_cls.__bases__:
                base_cls = base_cls.__bases__[0]
            else:
                raise ValueError("unable to find instance type")  # pragma: no cover

        return base_cls.__annotations__["instance_type"]


__all__ = ["BuildableConfig", "Exclude"]
