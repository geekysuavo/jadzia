"""Training functionality."""

# Standard
from typing import NamedTuple, Self, cast

# Third party
import equinox as eqx
import jax
from grain import DataLoader
from optax import GradientTransformation, OptState, Params
from orbax.checkpoint import v1 as ocp

# Dax
import jadzia.engine.logging as log
from jadzia.core.dependency.identifier import Reserved
from jadzia.engine.checkpointing import Checkpoint
from jadzia.engine.testing import TestArgs
from jadzia.nn.loss.composite import CompositeLoss
from jadzia.nn.model import CompositeModel


def compute_loss(
    model: CompositeModel,
    state: eqx.nn.State,
    loss_fn: CompositeLoss,
    xy: dict[str, jax.Array],
) -> tuple[jax.Array, eqx.nn.State]:
    """
    Compute the value of the loss function on a batch of data.

    Args:
        model: Current model to evaluate.
        state: Current model state.
        loss_fn: Loss function.
        xy: Model input and loss reference values.

    Returns:
        Loss function value.
        Updated model state.
    """
    (yhat, state) = model(xy, state)
    loss = loss_fn(yhat, xy)
    return (loss, state)


@eqx.filter_jit
def apply_step(
    model: CompositeModel,
    model_state: eqx.nn.State,
    loss_fn: CompositeLoss,
    optimizer: GradientTransformation,
    optimizer_state: OptState,
    xy: dict[str, jax.Array],
) -> tuple[CompositeModel, eqx.nn.State, OptState]:
    """
    Apply a single optimization step.

    Args:
        model: Current model to evaluate.
        model_state: Current model state.
        loss_fn: Loss function.
        optimizer: Optimizer.
        optimizer_state: Current optimizer state.
        xy: Model input and loss reference values.

    Returns:
        Updated model.
        Updated model state.
        Updated optimizer state.
    """
    params = cast(Params, model)
    compute_grad = eqx.filter_grad(compute_loss, has_aux=True)
    (grads, model_state) = compute_grad(model, model_state, loss_fn, xy)
    (updates, optimizer_state) = optimizer.update(grads, optimizer_state, params)
    model = eqx.apply_updates(model, updates)
    return (model, model_state, optimizer_state)


def train(
    model: CompositeModel,
    model_state: eqx.nn.State,
    loss_fn: CompositeLoss,
    optimizer: GradientTransformation,
    optimizer_state: OptState,
    dataloader: DataLoader,
    num_steps: int,
    checkpointer: ocp.training.Checkpointer | None = None,
    test_args: TestArgs | None = None,
    test_every: int = 0,
    log_every: int = 0,
) -> tuple[CompositeModel, eqx.nn.State, OptState]:
    """
    Train a model.

    Args:
        model: Model to train.
        model_state: Initial model state.
        loss_fn: Loss function.
        optimizer: Optimizer.
        optimizer_state: Initial optimizer state.
        dataloader: Data loader.
        num_steps: Number of training steps.
        checkpointer: Optional checkpointer.
        test_args: Optional testing function arguments.
        test_every: How often (in training steps) to run testing.
        log_every: How often (in training steps) to log loss values.

    Returns:
        Updated model.
        Updated model state.
        Updated optimizer state.
    """
    dataiter = iter(dataloader)
    begin_step = 0

    reloaded = Checkpoint(
        step=begin_step,
        model=model,
        model_state=model_state,
        dataloader_iter=dataiter,
        optimizer_state=optimizer_state,
    ).load(checkpointer)

    if reloaded:
        begin_step = reloaded.step + 1
        model = reloaded.model
        model_state = reloaded.model_state
        dataiter = reloaded.dataloader_iter
        optimizer_state = reloaded.optimizer_state
        assert checkpointer is not None
        log.info(f"[step={begin_step}] Loaded checkpoint from {checkpointer.directory}")

    for step in range(begin_step, num_steps):
        try:
            batch = next(dataiter)
        except StopIteration:
            dataiter = iter(dataloader)
            batch = next(dataiter)

        print(batch)#FIXME
        (model, model_state, optimizer_state) = apply_step(
            model,
            model_state,
            loss_fn,
            optimizer,
            optimizer_state,
            batch,
        )

        saved = Checkpoint(
            step=step,
            model=model,
            model_state=model_state,
            dataloader_iter=dataiter,
            optimizer_state=optimizer_state,
        ).save(checkpointer)

        if saved:
            assert checkpointer is not None
            log.info(f"[{step=}] Saved checkpoint to {checkpointer.directory}")

        if log_every and step % log_every == 0:
            (loss, _) = compute_loss(model, model_state, loss_fn, batch)
            log.info(f"[{step=}] Train loss={{loss:.6f}}", loss=loss, jax_async=True)

        if test_args and test_every and step % test_every == 0:
            (test_loss, test_terms) = test_args.update(model, model_state).run()
            test_terms = {Reserved.LOSS: test_loss, **test_terms}
            del test_loss

            fmt = ", ".join(f"{key}={{{key}:.6f}}" for key in test_terms)
            log.info(f"[{step=}] Test " + fmt, **test_terms, jax_async=True)

    return (model, model_state, optimizer_state)


class TrainArgs(NamedTuple):
    """Encapsulation of arguments to [jadzia.engine.training.train][]."""

    model: CompositeModel
    model_state: eqx.nn.State
    loss_fn: CompositeLoss
    optimizer: GradientTransformation
    optimizer_state: OptState
    dataloader: DataLoader
    num_steps: int
    checkpointer: ocp.training.Checkpointer | None

    test_args: TestArgs | None
    test_every: int
    log_every: int

    def update(
        self,
        model: CompositeModel,
        model_state: eqx.nn.State,
        optimizer_state: OptState,
    ) -> Self:
        """Return new arguments with updated an model and optimizer state."""
        if self.test_args is not None:
            test_args = self.test_args.update(model, model_state)
        else:
            test_args = None

        cls = type(self)
        return cls(
            model=model,
            model_state=model_state,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            optimizer_state=optimizer_state,
            dataloader=self.dataloader,
            num_steps=self.num_steps,
            checkpointer=self.checkpointer,
            test_args=test_args,
            test_every=self.test_every,
            log_every=self.log_every,
        )

    def run(self) -> Self:
        """Run training with the specified arguments."""
        return self.update(*train(*self))


__all__ = ["TrainArgs", "train"]
