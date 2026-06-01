"""Extensions to `yaml` format data."""

# Third party
import yaml
from yaml import dump, safe_load

# Dax
from jadzia.core.dependency.identifier import Identifier


def _construct_identifier(loader: yaml.SafeLoader, node: yaml.Node) -> Identifier:
    """Construct an identifier from a `yaml` node (originates from a `!batch` tag)."""
    value = loader.construct_scalar(node)
    return Identifier.from_str(value, is_batch=True)


def _represent_identifier(dumper: yaml.Dumper, ident: Identifier) -> yaml.Node:
    """Convert an identifier into a `yaml` node."""
    if ident.is_batch:
        return dumper.represent_scalar("!batch", str(ident))
    else:
        return dumper.represent_data(str(ident))


yaml.SafeLoader.add_constructor("!batch", _construct_identifier)
yaml.add_representer(Identifier, _represent_identifier)

__all__ = ["dump", "safe_load"]
