"""Composite model."""

# Standard
from collections.abc import Callable
from typing import NamedTuple, cast

# Third party
import equinox as eqx
import jax


class CompositeModelBlock(NamedTuple):
    """
    Details about a block in a composite model.

    Attributes:
        name: Name of the module asociated with the block.
        input_names: Other block outputs passed into the block.
        output_names: Names of outputs returned by the block.
    """

    name: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]


class CompositeModel(eqx.Module):
    """
    Directed acyclic graph of blocks.

    Attributes:
        modules: Submodules for each block.
        blocks: Details about each block.
        outputs: Names and block outputs of each model output.
    """

    class ReturnType(NamedTuple):
        """
        Model outputs and updated state.

        Attributes:
            outputs: Model outputs.
            state: Updated state.
        """

        outputs: dict[str, jax.Array]
        state: eqx.nn.State

    modules: dict[str, eqx.Module]
    blocks: tuple[CompositeModelBlock, ...]
    outputs: tuple[str, ...]

    def __call__(self, inputs: dict[str, jax.Array], state: eqx.nn.State) -> ReturnType:
        """
        Compute all blocks in their defined order.

        Args:
            inputs: Dictionary of input values.
            state: Input model state.

        Returns:
            Model outputs and updated model state.
        """
        arrays = inputs.copy()
        for block in self.blocks:
            module_inputs = (arrays[name] for name in block.input_names)

            module = cast(Callable, self.modules[block.name])
            if isinstance(module, eqx.nn.StatefulLayer) and module.is_stateful():
                (*module_outputs, state) = module(*module_inputs, state=state)
            else:
                module_outputs = module(*module_inputs)

            if isinstance(module_outputs, jax.Array):
                module_outputs = (module_outputs,)

            updates = dict(zip(block.output_names, module_outputs, strict=True))
            arrays.update(updates)

        outputs = {output: arrays[output] for output in self.outputs}
        return self.ReturnType(outputs, state)


__all__ = ["CompositeModel", "CompositeModelBlock"]
