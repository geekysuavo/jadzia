"""Tests of `yaml` extensions."""

# Third party
import pytest

# Dax
import jadzia.core.yaml as yaml
from jadzia.core.dependency.identifier import Identifier


@pytest.mark.parametrize("is_batch", [False, True])
def test_identifier_extension(tmp_path, is_batch):
    """Identifiers are convertible to and from `yaml` format."""
    ident = Identifier("hello", "world", is_batch)
    data = {"ident": ident}

    tmp_file = tmp_path / "ident.yaml"
    with tmp_file.open("w") as fh:
        yaml.dump(data, fh)

    has_batch_tag = "!batch" in tmp_file.read_text()
    with tmp_file.open() as fh:
        new_data = yaml.safe_load(fh)

    if is_batch:
        assert has_batch_tag is True
        assert new_data == data
    else:
        assert has_batch_tag is False
        assert new_data == {"ident": str(ident)}
