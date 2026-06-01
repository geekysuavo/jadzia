"""Composite loss function."""

# Standard
from collections.abc import Callable
from typing import NamedTuple

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp

# Dax
from jadzia.core.dependency.identifier import Identifier
from jadzia.nn.loss.masking import apply_mask, build_mask


class CompositeLossTerm(NamedTuple):
    """
    Details about a term in a composite loss function.

    Attributes:
        module: Module implementing the loss term.
        weight: Weight of the loss term in the linear combination.
        prediction_names: Names of predicted values required by the term.
        reference_names: Names of reference values required by the term.
    """

    module: Callable[..., jax.Array]
    weight: float
    prediction_names: tuple[Identifier, ...]
    reference_names: tuple[Identifier, ...]

    def __call__(
        self,
        predictions: dict[Identifier, jax.Array],
        references: dict[Identifier, jax.Array],
    ) -> jax.Array:
        """
        Compute the loss term value, including the weight.

        The finiteness pattern in the reference values is used to determine
        which values to use in the loss calculation. For a given batch item
        (first dimension) to be included, all elements along remaining
        dimensions must be finite.

        Args:
            predictions: Dictionary of predicted values.
            references: Dictionary of reference values.

        Returns:
            $()$, Weighted loss term value.
        """
        preds = [predictions[name] for name in self.prediction_names]
        refs = [references[name] for name in self.reference_names]

        if not refs:
            return self.weight * self.module(*preds).mean()

        ref_masks = [build_mask(ref) for ref in refs]
        finite_mask = jnp.stack(ref_masks, axis=0).all(axis=0)
        finite_weight = 1 / jnp.clip(finite_mask.sum(), min=1)

        preds_finite = (apply_mask(pred, finite_mask) for pred in preds)
        refs_finite = (apply_mask(ref, finite_mask) for ref in refs)

        loss = self.module(*preds_finite, *refs_finite)
        loss_finite = apply_mask(loss, finite_mask)

        return self.weight * finite_weight * loss_finite.sum()


class CompositeLoss(eqx.Module):
    """
    Linear combination of loss function terms.

    Attributes:
        terms: Details about each term in the sum.
    """

    terms: tuple[CompositeLossTerm, ...]

    def __call__(
        self,
        predictions: dict[Identifier, jax.Array],
        references: dict[Identifier, jax.Array],
    ) -> jax.Array:
        """
        Compute and combine all loss term values.

        Args:
            predictions: Dictionary of predicted values.
            references: Dictionary of reference values.

        Returns:
            $()$, Composite loss function value.
        """
        losses = [term(predictions, references) for term in self.terms]
        return jnp.stack(losses).sum()


__all__ = ["CompositeLoss", "CompositeLossTerm"]
