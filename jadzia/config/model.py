"""Composite model configuration."""

# Standard
from typing import ClassVar, Iterable, Literal, Self, overload

# Third party
import equinox as eqx
import jax
import jax.random as jr
from pydantic import model_validator

# Dax
from jadzia.config.block import Block
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.dependency.utility import sort_gathers
from jadzia.core.validation.block import Virtual
from jadzia.core.validation.mapping import NodeMappingConfig
from jadzia.nn.model import CompositeModel as CompositeModelImpl
from jadzia.nn.model import CompositeModelBlock


class CompositeModel(NodeMappingConfig[Block]):
    """Configuration for [jadzia.nn.model.CompositeModel][]."""

    instance_type: ClassVar[type[eqx.Module]] = CompositeModelImpl

    @property
    def requires_key(self) -> bool:
        """Whether any model building blocks requires a key."""
        return any(block.requires_key for block in self.values())

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: Literal[False] = False,
    ) -> CompositeModelImpl: ...

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: Literal[True] = True,
    ) -> tuple[CompositeModelImpl, eqx.nn.State]: ...

    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: Iterable[Identifier],
        with_state: bool = False,
    ) -> CompositeModelImpl | tuple[CompositeModelImpl, eqx.nn.State]:
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

        def dereference_virtual(name: str) -> str:
            """Dereference virtual block names."""
            block = self[name]
            if isinstance(block, Virtual):
                return block.ref
            else:
                return name

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
            if isinstance(block, Virtual):
                pass
            elif block.requires_key:
                modules[name] = block.build_instance(keys.pop())
            else:
                modules[name] = block.build_instance()

        blocks = tuple(
            CompositeModelBlock(
                name=name,
                input_names=tuple(map(str, self[name].input_names)),
                output_names=tuple(map(str, Identifier.from_outputs(name, self[name]))),
            )
            for name in map(dereference_virtual, order)
        )

        kwargs = {
            "modules": modules,
            "blocks": blocks,
            "outputs": tuple(map(str, outputs)),
        }

        if with_state:
            return eqx.nn.make_with_state(CompositeModelImpl)(**kwargs)
        else:
            return CompositeModelImpl(**kwargs)

    @model_validator(mode="after")
    def _valid_source_blocks(self) -> Self:
        """Ensure all virtual blocks refer to defined source blocks."""
        refs = {block.ref for block in self.values() if isinstance(block, Virtual)}
        if missing := refs - set(self):
            raise ValueError(f"reference blocks are missing: {', '.join(missing)}")
        else:
            return self


__all__ = ["CompositeModel"]
