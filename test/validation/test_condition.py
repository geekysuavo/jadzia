"""Tests of conditions and for-each expressions."""

# Third party
import pytest
from pydantic import TypeAdapter

# Dax
from jadzia.core.validation.condition import (
    Condition,
    ConditionOrForEachStr,
    ConditionStr,
    ForEach,
    Relation,
)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (2, 3),
        (5, 3),
        (7, 7),
        (1.5, 1.5),
        (2.1, 2.2),
        (8.8, 4.8),
        (True, True),
        (True, False),
        (False, False),
        ("foo", "bar"),
        ("hello", "hello"),
    ],
)
@pytest.mark.parametrize("relation", list(Relation))
def test_relation_eval(relation, a, b):
    """Relations encapsulate comparison operators."""
    expected = eval(f"{a!r} {relation.value} {b!r}")
    assert relation(a, b) is expected


def test_string_conversion():
    """Conditions and for-each expressions may be converted from strings."""
    condition = Condition.from_str("foo == 42")
    assert condition.name == "foo"
    assert condition.relation == Relation.EQ
    assert condition.value == 42

    for_each = ForEach.from_str("for-each var")
    assert for_each.name == "var"

    with pytest.raises(TypeError):
        Condition.from_str(7)

    with pytest.raises(TypeError):
        ForEach.from_str(7)

    with pytest.raises(ValueError, match="invalid condition expression"):
        Condition.from_str("invalid-name == 42")

    with pytest.raises(ValueError, match="Input should be"):
        Condition.from_str("bar << 42")

    with pytest.raises(ValueError, match="invalid for-each expression"):
        ForEach.from_str("for-each !")

    with pytest.raises(ValueError, match="invalid for-each expression"):
        ForEach.from_str("for-all zaz")


def test_conversion_to_string():
    """Conditions and for-each expressions may be converted to strings."""
    condition = Condition(name="x", relation=Relation.LT, value=5.9)
    assert str(condition) == "x < 5.9"

    for_each = ForEach(name="y")
    assert str(for_each) == "for-each y"


def test_condition_eval():
    """Conditions may be called on mappings to return a boolean mask."""
    data = {"foo": 1.0}

    condition = Condition(name="foo", relation="<", value=0)
    assert condition(data) is False

    condition = Condition(name="foo", relation=">", value=0)
    assert condition(data) is True

    other_data = {"bar": 1.0}
    with pytest.raises(KeyError, match="foo"):
        condition(other_data)


@pytest.mark.parametrize("allow_for_each", [False, True])
def test_type_annotations(allow_for_each):
    """Condition and for-each expression type annotations behave as expected."""
    if allow_for_each:
        adapter = TypeAdapter(ConditionOrForEachStr)
    else:
        adapter = TypeAdapter(ConditionStr)

    condition = adapter.validate_python('a != "b"')
    condition_str = adapter.dump_python(condition)
    assert condition == Condition(name="a", relation=Relation.NE, value="b")
    assert condition_str == 'a != "b"'

    if not allow_for_each:
        with pytest.raises(ValueError, match="invalid condition expression"):
            adapter.validate_python("for-each seven")
        return

    for_each = adapter.validate_python("for-each seven")
    for_each_str = adapter.dump_python(for_each)
    assert for_each == ForEach(name="seven")
    assert for_each_str == "for-each seven"

    with pytest.raises(ValueError, match="invalid condition or for-each expression"):
        adapter.validate_python("just really wrong")
