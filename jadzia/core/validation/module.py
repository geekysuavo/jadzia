"""Validation base class for buildable modules."""

# Standard
from typing import Any, ClassVar, Literal, Self, overload

# Third party
import equinox as eqx
import jax

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.buildable import BuildableConfig
from jadzia.core.validation.inspect import inspect_callable

# reserved parameter names.
_KEY: str = "key"
_STATE: str = "state"


class ModuleSignature(Config):
    """
    Call method signature of a module.

    Attributes:
        inputs_with_defaults: Names of input parameters that have default values.
        inputs_pos_only: Names of input parameters that must be passed positionally.
        inputs_pos_or_kw: Names of input parameters that may be passed either
            positionally or by keyword.
        inputs_kw_only: Names of input parameters that must be passed by keyword.
        outputs: Names of output values, or `output` when the callable returns
            a single output.
    """

    inputs_with_defaults: tuple[str, ...]
    inputs_pos_only: tuple[str, ...]
    inputs_pos_or_kw: tuple[str, ...]
    inputs_kw_only: tuple[str, ...]
    outputs: tuple[str, ...]

    @classmethod
    def from_type(cls, module_type: type[eqx.Module]) -> Self:
        """
        Inspect a module type to determine its call signature.

        Args:
            module_type: Type of module to inspect.

        Returns:
            Module call method signature.
        """
        names = inspect_callable(module_type.__call__)
        return cls(
            inputs_with_defaults=cls._filter_state(names.inputs_with_defaults),
            inputs_pos_only=cls._filter_state(names.inputs_pos_only),
            inputs_pos_or_kw=cls._filter_state(names.inputs_pos_or_kw),
            inputs_kw_only=cls._filter_state(names.inputs_kw_only),
            outputs=names.outputs,
        )

    @staticmethod
    def _filter_state(names: tuple[str, ...]) -> tuple[str, ...]:
        """Filter out the reserved `state` parameter name from an input sequence."""
        return tuple(filter(lambda name: name != _STATE, names))


class ModuleConfig(BuildableConfig[eqx.Module]):
    """Base class for all validated module configurations."""

    instance_type: ClassVar[type[eqx.Module]]
    instance_call: ClassVar[ModuleSignature]
    requires_key: ClassVar[bool] = False

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        with_state: Literal[False] = False,
    ) -> eqx.Module: ...

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        with_state: Literal[True] = True,
    ) -> tuple[eqx.Module, eqx.nn.State]: ...

    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        with_state: bool = False,
    ) -> eqx.Module | tuple[eqx.Module, eqx.nn.State]:
        """
        Build an instance based on this configuration.

        Args:
            key: Pseudorandom number generator key. Must only be provided when
                building instances of modules that require a key.
            with_state: When true, build the module with its initial state.

        Returns:
            Newly built module instance, with an optional initial state.
        """
        kwargs = self.model_dump_for_instance(pass_subclasses=(ModuleConfig,))
        name = self.instance_type.__name__
        cls = type(self)

        if cls.requires_key and key is None:
            raise ValueError(f"module {name} requires a key")
        elif not cls.requires_key and key is not None:
            raise ValueError(f"module {name} does not require a key")
        elif key is not None:
            kwargs[_KEY] = key

        if with_state:
            return eqx.nn.make_with_state(cls.instance_type)(**kwargs)
        else:
            return cls.instance_type(**kwargs)

    @classmethod
    def auto_hook(
        cls,
        instance_type: type[eqx.Module],
        *,
        field_defs: dict[str, tuple[Any, Any]],
    ):
        """Modify the field definitions obtained by inspection."""
        # determine the module call signature.
        field_defs["instance_call"] = (
            ClassVar[ModuleSignature],
            ModuleSignature.from_type(instance_type),
        )

        # determine whether the module initializer requires a key.
        field_defs["requires_key"] = (ClassVar[bool], _KEY in field_defs)
        field_defs.pop(_KEY, None)


__all__ = ["ModuleConfig", "ModuleSignature"]
