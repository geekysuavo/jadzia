"""Testing functionality."""

# Standard
import itertools
from typing import NamedTuple, Self

# Third party
import equinox as eqx
import jax
from grain import DataLoader

# Dax
from jadzia.engine.averaging import Average
from jadzia.engine.utility import make_send_to_device
from jadzia.nn.loss.composite import CompositeLoss
from jadzia.nn.model import CompositeModel


@eqx.filter_jit
def apply_step(
    model: CompositeModel,
    model_state: eqx.nn.State,
    loss_fn: CompositeLoss,
    loss_state: Average[CompositeLoss.ReturnType] | None,
    xy: dict[str, jax.Array],
) -> Average[CompositeLoss.ReturnType]:
    """
    Apply a single loss averaging step.

    Args:
        model: Model to evaluate.
        model_state: Model state.
        loss_fn: Loss function.
        loss_state: Current loss average, or `None` to begin averaging.
        xy: Model input and loss reference values.

    Returns:
        Updated loss average.
        Number of averaged loss values.
    """
    (yhat, _) = model(xy, state=model_state)
    new_state = loss_fn(yhat, xy, return_terms=True)
    if loss_state is None:
        return Average[CompositeLoss.ReturnType](new_state)
    else:
        return loss_state.update(new_state)


def test(
    model: CompositeModel,
    model_state: eqx.nn.State,
    loss_fn: CompositeLoss,
    dataloader: DataLoader,
    sharding: jax.sharding.Sharding,
    num_steps: int | None,
) -> CompositeLoss.ReturnType:
    """
    Test a model.

    Args:
        model: Model to evaluate.
        model_state: Model state.
        loss_fn: Loss function.
        dataloader: Data loader.
        sharding: Batch data sharding.
        num_steps: Number of training steps per epoch.

    Returns:
        Loss values averaged over all batches.
    """
    if num_steps:
        dataiter = itertools.islice(dataloader, num_steps)
    else:
        dataiter = dataloader

    loss_state = None
    send_to_device = make_send_to_device(sharding)
    for step, batch in enumerate(map(send_to_device, dataiter)):
        loss_state = apply_step(model, model_state, loss_fn, loss_state, batch)

    assert loss_state is not None
    return loss_state.value


class TestArgs(NamedTuple):
    """Encapsulation of arguments to [jadzia.engine.testing.test][]."""

    __test__ = False

    model: CompositeModel
    model_state: eqx.nn.State
    loss_fn: CompositeLoss
    dataloader: DataLoader
    sharding: jax.sharding.Sharding
    num_steps: int | None

    def update(self, model: CompositeModel, model_state: eqx.nn.State) -> Self:
        """Return new arguments with updated an model and model state."""
        cls = type(self)
        return cls(
            model=model,
            model_state=model_state,
            loss_fn=self.loss_fn,
            dataloader=self.dataloader,
            sharding=self.sharding,
            num_steps=self.num_steps,
        )

    def run(self) -> CompositeLoss.ReturnType:
        """Run testing with the specified arguments."""
        return test(*self)


__all__ = ["TestArgs", "test"]
