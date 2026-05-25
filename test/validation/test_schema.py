"""Tests of schema version validation."""

# Third party
import pytest

# Dax
from jadzia.core.validation.base import Config
from jadzia.core.validation.schema import SchemaVersion, SchemaVersionStr


class Example(Config):
    checked: SchemaVersionStr
    unchecked: SchemaVersion


@pytest.fixture
def versions():
    package_version = SchemaVersion.from_metadata()
    different_version = SchemaVersion(package_version.major, package_version.minor + 1)
    return (package_version, different_version)


@pytest.fixture
def version_strs(versions):
    return tuple(map(str, versions))


@pytest.mark.parametrize("minor", [3, 4, 5])
@pytest.mark.parametrize("major", [0, 1, 2])
def test_unchecked(major, minor):
    """Schema versions may be converted to and from strings."""
    version_str = f"{major}.{minor}"
    version = SchemaVersion.from_str(version_str, check=False)
    assert version == SchemaVersion(major, minor)
    assert str(version) == version_str
    assert version.major == major
    assert version.minor == minor

    with pytest.raises(TypeError, match="expected str"):
        SchemaVersion.from_str(1.2)

    for invalid_str in ("12", "1.2.3"):
        with pytest.raises(ValueError, match=r"expected '.*' format"):
            SchemaVersion.from_str(invalid_str)


def test_checked(versions, version_strs):
    """Schema versions are checked against the package version by default."""
    (package_version, different_version) = versions
    (package_version_str, different_version_str) = version_strs

    assert different_version != package_version
    assert different_version_str != package_version_str

    same_version = SchemaVersion.from_str(package_version_str)
    assert same_version == package_version

    with pytest.raises(ValueError, match="expected schema version"):
        SchemaVersion.from_str(different_version_str)


def test_validation(versions, version_strs):
    """Schema version strings are checked during validation."""
    (package_version, different_version) = versions
    (package_version_str, different_version_str) = version_strs

    example = Example(unchecked=different_version, checked=package_version_str)
    assert example.unchecked == different_version
    assert example.checked == package_version

    with pytest.raises(ValueError, match="must be a tuple"):
        Example(unchecked=different_version_str, checked=package_version_str)

    with pytest.raises(ValueError, match="expected schema version"):
        Example(unchecked=different_version, checked=different_version_str)
