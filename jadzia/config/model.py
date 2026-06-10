"""Composite model configuration."""

# Standard
from typing import ClassVar, Iterable, Literal, overload

# Third party
import equinox as eqx
import jax
import jax.random as jr

# Dax
from jadzia.config.block import Block
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.utility import sort_gathers
from jadzia.core.validation.mapping import NodeMappingConfig
from jadzia.nn.model import CompositeModel as CompositeModelImpl
from jadzia.nn.model import CompositeModelBlock


class CompositeModel(NodeMappingConfig[Block]):
    """Configuration for [jadzia.nn.model.CompositeModel][]."""

    instance_type: ClassVar[type[eqx.Module]] = CompositeModelImpl

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: Literal[False] = False,
    ) -> eqx.Module: ...

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: Literal[True] = True,
    ) -> tuple[eqx.Module, eqx.nn.State]: ...

    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: bool = False,
    ) -> eqx.Module | tuple[eqx.Module, eqx.nn.State]:
        """
        Build an instance based on this configuration.

        Args:
            key: Pseudorandom number generator key. Must only be provided when
                one or more model building blocks requires a key to be built.
            outputs: Iterable of model outputs to include.
            with_state: When true, build the module with its initial state.

        Returns:
            Newly built module instance, with an optional initial state.
        """
        num_keys = sum(block.requires_key for block in self.values())
        if num_keys and key is None:
            raise ValueError("module requires a key")
        elif not num_keys and key is not None:
            raise ValueError("module does not require a key")

        keys = list(jr.split(key, num=num_keys)) if key is not None else []
        del key

        order = sort_gathers(self.build_subgraph(), subset=outputs)

        modules = {}
        for name in order:
            block = self[name]
            if block.requires_key:
                modules[name] = block.build_instance(keys.pop())
            else:
                modules[name] = block.build_instance()

        blocks = tuple(
            CompositeModelBlock(
                name=name,
                input_names=tuple(map(str, self[name].input_names)),
                output_names=tuple(map(str, Identifier.from_outputs(name, self[name]))),
            )
            for name in order
        )
        kwargs = {
            "modules": modules,
            "blocks": blocks,
            "outputs": tuple(map(str, outputs)),
        }

        cls = type(self)
        if with_state:
            return eqx.nn.make_with_state(cls.instance_type)(**kwargs)
        else:
            return cls.instance_type(**kwargs)


__all__ = ["CompositeModel"]
