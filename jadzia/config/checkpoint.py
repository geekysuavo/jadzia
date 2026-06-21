"""Checkpointer configuration."""

# Standard
from typing import cast

# Third party
from orbax.checkpoint import v1 as ocp
from pydantic import NonNegativeInt, PositiveInt

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.context import WithRuntimeContext

save = ocp.training.save_decision_policies
preserve = ocp.training.preservation_policies


class CheckpointConfig(Config, WithRuntimeContext):
    """
    Checkpointer configuration.

    Attributes:
        every: How often (in steps) to save checkpoints.
        retain: Minimum number of checkpoints to preserve.
        on_steps: Steps for which checkpoints should be saved and preserved.
        on_preemption: Whether checkpoints should be saved on preemption.
        on_start: Whether checkpoints should be saved on startup.
    """

    every: PositiveInt | None = None
    retain: PositiveInt | None = None
    on_steps: tuple[NonNegativeInt, ...] | None = None
    on_preemption: bool = False
    on_start: bool = False

    def build_instance(self) -> ocp.training.Checkpointer:
        """Build an instance from this configuration."""
        save_policies = []
        preserve_policies = []

        if self.every:
            save_policies.append(save.FixedIntervalPolicy(interval=self.every))

        if self.on_steps:
            save_policies.append(save.SpecificStepsPolicy(steps=self.on_steps))
            preserve_policies.append(preserve.CustomSteps(steps=self.on_steps))

        if self.on_preemption:
            save_policies.append(save.PreemptionCheckpointingPolicy())

        if self.on_start:
            save_policies.append(save.InitialSavePolicy())

        if self.retain:
            preserve_policies.append(preserve.LatestN(n=self.retain))

        if save_policies:
            save_decision_policy = cast(
                save.SaveDecisionPolicy,
                save.AnySavePolicy(policies=tuple(save_policies)),
            )
        else:
            save_decision_policy = None

        if preserve_policies:
            preservation_policy = cast(
                preserve.PreservationPolicy,
                preserve.AnyPreservationPolicy(policies=tuple(preserve_policies)),
            )
        else:
            preservation_policy = None

        return ocp.training.Checkpointer(
            directory=str(self.context.checkpoint_dir.resolve()),
            save_decision_policy=save_decision_policy,
            preservation_policy=preservation_policy,
        )


__all__ = ["CheckpointConfig"]
