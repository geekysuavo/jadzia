"""Validation base classes for transformations."""

# Standard
from typing import Annotated, Any, ClassVar, Self

# Third party
from pydantic import model_validator

# Dax
from jadzia.core.dependency.identifier import Identifier, IdentifierStr
from jadzia.core.validation.base import Config
from jadzia.core.validation.buildable import BuildableConfig, Exclude
from jadzia.core.validation.inspect import inspect_callable
from jadzia.data.transform.base import Transformation

# reserved parameter names.
_INDEX: str = "index"
_RNG: str = "rng"


class TransformSignature(Config):
    """
    Call method signature of a transformation.

    Attributes:
        inputs: Names of positional-or-keyword input parameters.
        outputs: Names of output values, or `output` when the transform
            returns a single output.
        requires_index: Whether the transform is indexed.
        requires_rng: Whether the transform is random.
    """

    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    requires_index: bool
    requires_rng: bool

    @classmethod
    def from_type(cls, transform_type: type[Transformation]) -> Self:
        """
        Inspect a transformation type to determine its call signature.

        Args:
            transform_type: Type of transformation to inspect.

        Returns:
            Transformation call method signature.
        """
        names = inspect_callable(transform_type.__call__)
        if _INDEX in names.inputs_pos_only + names.inputs_pos_or_kw:
            raise ValueError("index parameter must be keyword-only")
        elif _RNG in names.inputs_pos_only + names.inputs_pos_or_kw:
            raise ValueError("rng parameter must be keyword-only")

        return cls(
            inputs=names.inputs_pos_only + names.inputs_pos_or_kw,
            outputs=names.outputs,
            requires_index=_INDEX in names.inputs_kw_only,
            requires_rng=_RNG in names.inputs_kw_only,
        )


class TransformConfig(BuildableConfig[Transformation]):
    """
    Transformation configuration base class.

    Attributes:
        inputs: Mapping from transform call method argument names to input names
            in `tform.output` format.
    """

    instance_type: ClassVar[type[Transformation]]
    instance_call: ClassVar[TransformSignature]

    inputs: Annotated[dict[str, IdentifierStr], Exclude]

    def build_instance(  # ty: ignore[invalid-method-override]
        self,
        name: str,
    ) -> Transformation:
        """
        Build an instance based on this configuration.

        Args:
            name: Name of the transformation.

        Returns:
            Newly built transformation instance.
        """
        cls = type(self)
        kwargs = self.model_dump_for_instance()
        input_names = tuple(map(str, self.input_names))
        output_names = tuple(map(str, Identifier.from_outputs(name, self)))
        return cls.instance_type(
            **kwargs,
            input_names=input_names,
            output_names=output_names,
        )

    @property
    def input_names(self) -> tuple[Identifier, ...]:
        """Names of other outputs that are passed as inputs to this transformation."""
        original_names = self.instance_call.inputs
        return tuple(self.inputs[name] for name in original_names)

    @property
    def output_names(self) -> tuple[str, ...]:
        """Names of outputs that are returned by this transformation."""
        return self.instance_call.outputs

    @classmethod
    def auto_hook(
        cls,
        instance_type: type[Transformation],
        *,
        field_defs: dict[str, tuple[Any, Any]],
    ):
        """Modify the field definitions obtained by inspection."""
        # determine the transform call signature.
        field_defs["instance_call"] = (
            ClassVar[TransformSignature],
            TransformSignature.from_type(instance_type),
        )

        # remove fields provided by this class.
        field_defs.pop("input_names")
        field_defs.pop("output_names")

    @model_validator(mode="after")
    def _validate_input_names(self) -> Self:
        """Ensure the required number and kind of inputs is provided."""
        supplied_names = set(self.inputs)
        required_names = set(self.instance_call.inputs)
        missing_names = required_names - supplied_names
        extra_names = supplied_names - required_names
        if missing_names:
            raise ValueError(
                f"missing transformation call inputs in {type(self).__name__}: "
                f"({', '.join(missing_names)})"
            )
        elif extra_names:
            raise ValueError(
                f"extra transformation call inputs in {type(self).__name__}: "
                f"({', '.join(extra_names)})"
            )
        else:
            return self


__all__ = ["TransformConfig", "TransformSignature"]
