"""Composite loss function."""

# Standard
from collections.abc import Callable
from typing import Literal, NamedTuple, overload

# Third party
import equinox as eqx
import jax
import jax.numpy as jnp

# Dax
from jadzia.nn.loss.masking import apply_mask, build_mask


class CompositeLossTerm(NamedTuple):
    """
    Details about a term in a composite loss function.

    Attributes:
        module: Module implementing the loss term.
        name: Name of the loss term.
        weight: Weight of the loss term in the linear combination.
        prediction_names: Names of predicted values required by the term.
        reference_names: Names of reference values required by the term.
    """

    module: Callable[..., jax.Array]
    name: str
    weight: float
    prediction_names: tuple[str, ...]
    reference_names: tuple[str, ...]

    def __call__(
        self,
        predictions: dict[str, jax.Array],
        references: dict[str, jax.Array],
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

    class ReturnType(NamedTuple):
        """
        Loss value and per-term values.

        Attributes:
            value: Composite loss function value.
            terms: Dictionary of per-term values.
        """

        value: jax.Array
        terms: dict[str, jax.Array]

    terms: tuple[CompositeLossTerm, ...]

    @overload
    def __call__(
        self,
        predictions: dict[str, jax.Array],
        references: dict[str, jax.Array],
        *,
        return_terms: Literal[False] = False,
    ) -> jax.Array: ...

    @overload
    def __call__(
        self,
        predictions: dict[str, jax.Array],
        references: dict[str, jax.Array],
        *,
        return_terms: Literal[True] = True,
    ) -> jax.Array: ...

    def __call__(
        self,
        predictions: dict[str, jax.Array],
        references: dict[str, jax.Array],
        *,
        return_terms: bool = False,
    ) -> jax.Array | ReturnType:
        """
        Compute and combine all loss term values.

        Args:
            predictions: Dictionary of predicted values.
            references: Dictionary of reference values.
            return_terms: When true, additionally return per-term loss values.

        Returns:
            Composite loss function value, or the value with per-term values.
        """
        losses = [term(predictions, references) for term in self.terms]
        value = jnp.stack(losses).sum()
        if return_terms:
            terms = {term.name: loss for term, loss in zip(self.terms, losses)}
            return self.ReturnType(value, terms)
        else:
            return value


__all__ = ["CompositeLoss", "CompositeLossTerm"]
