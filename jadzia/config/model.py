"""Composite model configuration."""

# Standard
from typing import Annotated, ClassVar, Literal, Self, overload

# Third party
import equinox as eqx
import jax
import jax.random as jr
from pydantic import Field, model_validator

# Dax
from jadzia.config.block import Block
from jadzia.core.dependency.graph import Graph
from jadzia.core.dependency.identifier import Identifier, IdentifierStr
from jadzia.core.validation.block import BlockConfig
from jadzia.core.validation.module import ModuleSignature
from jadzia.nn.model import CompositeModel as CompositeModelImpl
from jadzia.nn.model import CompositeModelBlock


class CompositeModel(BlockConfig):
    """
    Configuration for [jadzia.nn.model.CompositeModel][].

    Attributes:
        outputs: Names of each model output mapped to their associated block outputs.
        blocks: Definitions of each model building block.
    """

    instance_type: ClassVar[type[eqx.Module]] = CompositeModelImpl
    instance_call: ClassVar[ModuleSignature] = ModuleSignature.from_type(
        CompositeModelImpl
    )

    outputs: Annotated[dict[str, IdentifierStr], Field(min_length=1)]
    blocks: dict[str, Block]

    @property
    def input_names(self) -> tuple[str, ...]:
        """Names of batch entries that are passed as inputs to the model."""

        def is_batch(output: Identifier) -> bool:
            return output.is_batch

        def to_str(output: Identifier) -> str:
            return output.output

        block_inputs = (block.input_names for block in self.blocks.values())
        batch_inputs = map(to_str, filter(is_batch, set().union(*block_inputs)))
        return tuple(sorted(batch_inputs))

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of outputs that are returned by the model."""
        return tuple(sorted(self.outputs))

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
                one or more model building blocks requires a key to be built.
            with_state: When true, build the module with its initial state.

        Returns:
            Newly built module instance, with an optional initial state.
        """
        num_keys = sum(block.requires_key for block in self.blocks.values())
        if num_keys and key is None:
            raise ValueError("module requires a key")
        elif not num_keys and key is not None:
            raise ValueError("module does not require a key")

        keys = list(jr.split(key, num=num_keys)) if key is not None else []
        del key

        modules = {}
        for name, block in self.blocks.items():
            if block.requires_key:
                modules[name] = block.build_instance(keys.pop(), with_state=with_state)
            else:
                modules[name] = block.build_instance(with_state=with_state)

        blocks = tuple(
            CompositeModelBlock(
                name=name,
                input_names=self.blocks[name].input_names,
                output_names=self.blocks[name].output_names,
            )
            for name in self._block_order()
        )
        outputs = tuple(self.outputs.items())
        kwargs = {"modules": modules, "blocks": blocks, "outputs": outputs}

        cls = type(self)
        if with_state:
            return eqx.nn.make_with_state(cls.instance_type)(**kwargs)
        else:
            return cls.instance_type(**kwargs)

    def _block_order(self) -> tuple[str, ...]:
        """Determine the execution order of each model building block."""
        deps = {
            name: {dep.node for dep in block.input_names}
            for name, block in self.blocks.items()
        }
        return Graph(deps).sort_nodes()

    @model_validator(mode="after")
    def _ensure_valid_model(self) -> Self:
        """Ensure that the model configuration is valid."""
        # condition: all non-batch block inputs are satisfied by other blocks.
        def not_batch(output: Identifier) -> bool:
            return not output.is_batch

        all_block_inputs = {
            name: set(filter(not_batch, block.input_names))
            for name, block in self.blocks.items()
        }

        all_block_outputs: set[Identifier] = set()
        for name, block in self.blocks.items():
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

        # condition: only blocks required to compute outputs are defined.
        outputs_to_build = set(self.outputs.values())
        built_block_names = set()
        while outputs_to_build:
            output = outputs_to_build.pop()
            if output.is_batch or output.node in built_block_names:
                continue

            name = output.node
            block = self.blocks[name]

            built_block_names.add(name)
            outputs_to_build.update(block.input_names)

        unused_blocks = set(self.blocks) - built_block_names
        if unused_blocks:
            raise ValueError(f"blocks are unused: {', '.join(sorted(unused_blocks))}")

        # condition: the block dependencies form a directed acyclic graph.
        self._block_order()
        return self


__all__ = ["CompositeModel"]
