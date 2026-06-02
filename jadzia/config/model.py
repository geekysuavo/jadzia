"""Composite model configuration."""

# Standard
from typing import ClassVar, Literal, Self, overload

# Third party
import equinox as eqx
import jax
import jax.random as jr
from pydantic import model_validator

# Dax
from jadzia.config.block import Block
from jadzia.core.dependency.identifier import Identifier
from jadzia.core.validation.mapping import NodeMappingConfig
from jadzia.nn.model import CompositeModel as CompositeModelImpl
from jadzia.nn.model import CompositeModelBlock


class CompositeModel(NodeMappingConfig[Block]):
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

        nodes = self.node_names(outputs)
        if nodes.unused:
            raise ValueError(f"blocks are unused: {', '.join(nodes.unused)}")

        blocks = tuple(
            CompositeModelBlock(
                name=name,
                input_names=tuple(map(str, self[name].input_names)),
                output_names=tuple(map(str, Identifier.from_outputs(name, self[name]))),
            )
            for name in self.node_order(nodes.used)
        )
        kwargs = {"modules": modules, "blocks": blocks, "outputs": outputs}

        cls = type(self)
        if with_state:
            return eqx.nn.make_with_state(cls.instance_type)(**kwargs)
        else:
            return cls.instance_type(**kwargs)

    @model_validator(mode="after")
    def _ensure_valid_model(self) -> Self:
        """Ensure that the model configuration is valid."""
        (_, _, missing) = self.node_names()
        missing = set(filter(lambda output: not output.is_batch, missing))
        if missing:
            error_lines = [
                "block inputs are not satisfied.",
                f"all block outputs: ({', '.join(map(str, self.output_names))})",
                f"missing inputs: ({', '.join(map(str, missing))})"
                "",
            ]
            raise ValueError("\n".join(error_lines))
        else:
            self.node_order()
            return self


__all__ = ["CompositeModel"]
