"""Composite model configuration."""

# Standard
from collections.abc import Iterable
from typing import ClassVar, Literal, Self, overload

# Third party
import equinox as eqx
import jax
import jax.random as jr
from pydantic import model_validator

# Dax
from jadzia.config.block import Block
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.mapping import MappingConfig
from jadzia.nn.model import CompositeModel as CompositeModelImpl
from jadzia.nn.model import CompositeModelBlock


class CompositeModel(MappingConfig[Block]):
    """Configuration for [jadzia.nn.model.CompositeModel][]."""

    instance_type: ClassVar[type[eqx.Module]] = CompositeModelImpl

    @property
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of batch entries that are passed as inputs to the model."""

        def is_batch(output: Identifier) -> bool:
            return output.is_batch

        block_inputs = (block.input_names for block in self.values())
        batch_inputs = filter(is_batch, set().union(*block_inputs))
        return tuple(sorted(batch_inputs))

    @property
    def output_names(self) -> tuple[Identifier, ...]:
        """Names of outputs that are returned by the model."""

        def to_identifiers(item: tuple[str, Block]) -> tuple[Identifier, ...]:
            return Identifier.from_outputs(*item)

        block_outputs = set().union(*map(to_identifiers, self.items()))
        return tuple(sorted(block_outputs))

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: tuple[Identifier, ...] = (),
        with_state: Literal[False] = False,
    ) -> eqx.Module: ...

    @overload
    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: tuple[Identifier, ...] = (),
        with_state: Literal[True] = True,
    ) -> tuple[eqx.Module, eqx.nn.State]: ...

    def build_instance(
        self,
        key: jax.Array | None = None,
        *,
        outputs: tuple[Identifier, ...] = (),
        with_state: bool = False,
    ) -> eqx.Module | tuple[eqx.Module, eqx.nn.State]:
        """
        Build an instance based on this configuration.

        Args:
            key: Pseudorandom number generator key. Must only be provided when
                one or more model building blocks requires a key to be built.
            outputs: Optional sequence of required model outputs. When left
                unspecified, all model outputs will be returned.
            with_state: When true, build the module with its initial state.

        Returns:
            Newly built module instance, with an optional initial state.
        """
        outputs = outputs or self.output_names
        missing_outputs = set(outputs) - set(self.output_names)
        if missing_outputs:
            missing_str = ", ".join(map(str, missing_outputs))
            raise ValueError(f"missing outputs: {missing_str}")

        num_keys = sum(block.requires_key for block in self.values())
        if num_keys and key is None:
            raise ValueError("module requires a key")
        elif not num_keys and key is not None:
            raise ValueError("module does not require a key")

        keys = list(jr.split(key, num=num_keys)) if key is not None else []
        del key

        modules = {}
        for name, block in self.items():
            if block.requires_key:
                modules[name] = block.build_instance(keys.pop())
            else:
                modules[name] = block.build_instance()

        blocks = tuple(
            CompositeModelBlock(
                name=name,
                input_names=self[name].input_names,
                output_names=self[name].output_names,
            )
            for name in self._block_order(self._block_names(outputs))
        )
        kwargs = {"modules": modules, "blocks": blocks, "outputs": outputs}

        cls = type(self)
        if with_state:
            return eqx.nn.make_with_state(cls.instance_type)(**kwargs)
        else:
            return cls.instance_type(**kwargs)

    def _block_names(self, outputs: Iterable[Identifier]) -> set[str]:
        """Determine which blocks are required to produce a set of outputs."""
        outputs_to_build = set(outputs)
        built_block_names = set()
        while outputs_to_build:
            output = outputs_to_build.pop()
            if output.is_batch or output.node in built_block_names:
                continue

            name = output.node
            block = self[name]

            built_block_names.add(name)
            outputs_to_build.update(block.input_names)

        unused_blocks = set(self) - built_block_names
        if unused_blocks:
            raise ValueError(f"blocks are unused: {', '.join(sorted(unused_blocks))}")
        else:
            return built_block_names

    def _block_order(self, subset: Iterable[str] | None = None) -> tuple[str, ...]:
        """Determine the execution order of each model building block."""
        deps = {
            name: {dep.node for dep in block.input_names}
            for name, block in self.items()
        }
        return Graph(deps).sort_nodes(subset)

    @model_validator(mode="after")
    def _ensure_valid_model(self) -> Self:
        """Ensure that the model configuration is valid."""

        # condition: all non-batch block inputs are satisfied by other blocks.
        def not_batch(output: Identifier) -> bool:
            return not output.is_batch

        all_block_inputs = {
            name: set(filter(not_batch, block.input_names))
            for name, block in self.items()
        }

        all_block_outputs: set[Identifier] = set()
        for name, block in self.items():
            all_block_outputs.update(Identifier.from_outputs(name, block))

        missing_block_inputs = {
            name: block_inputs - all_block_outputs
            for name, block_inputs in all_block_inputs.items()
            if block_inputs - all_block_outputs
        }

        if missing_block_inputs:
            missing_lines = [
                f"missing inputs to '{name}': ({', '.join(map(str, outputs))})"
                for name, outputs in missing_block_inputs.items()
            ]
            error_lines = [
                "block inputs are not satisfied.",
                f"all block outputs: ({', '.join(map(str, all_block_outputs))})",
                *missing_lines,
                "",
            ]
            raise ValueError("\n".join(error_lines))

        # condition: the block dependencies form a directed acyclic graph.
        self._block_order()
        return self


__all__ = ["CompositeModel"]
