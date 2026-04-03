from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset
    from typing import Any, Set, Sequence, Type

__all__ = [
    '_check_literal',
    '_check_columns',
    '_check_type',
    '_check_inner_type',
    '_check_only_one',
]


def _check_literal(name: str, value: Any, expected: Set[Any]) -> None:
    """
    Check that a value is in a set of expected literals.

    Parameters
    ----------
    name : str
        Name of the parameter being checked, used in error messages.
    value : Any
        Value of the parameter being checked.
    expected : Set[Any]
        Set of expected literal values.

    Raises
    ------
    ValueError
        Value was not in expected set of literals.

    """
    if value not in expected:
        raise ValueError(f"'{name}' must be one of {expected}, got {value}.")


def _check_columns(data: Dataset, columns: Sequence[str]) -> None:
    """
    Check that a dataset contains required columns.

    Parameters
    ----------
    data : Dataset
        A dataset whose columns will be checked against those required.
    columns : Sequence[str]
        A sequence of column names required for a processing function.

    Raises
    ------
    ValueError
        If the data is missing columns required for a processing function.

    """
    missing = list(set(columns) - set(data.columns))
    if missing:
        raise ValueError(f"'data' requires {columns=}, but {missing=}.")


def _check_type(name: str, value: Any, expected: Type | tuple[Type]) -> None:
    """
    Check the type of a parameter against expected type(s).

    Parameters
    ----------
    name : str
        Name of the parameter being checked, used in error messages.
    value : Any
        Value of the parameter being checked.
    expected : Type or tuple[Type]
        The expected type(s) for the parameter.

    Raises
    ------
    TypeError
        If the parameter is not one of the expected types.

    """
    if isinstance(expected, tuple):
        correction = [t if t is not None else type(None) for t in expected]
        exp_names = "(" + ", ".join(t.__name__ for t in correction) + ")"
    else:
        expected = (expected,)  # Ensure expected is a tuple for uniformity
        correction = [t if t is not None else type(None) for t in expected]
        exp_names = correction[0].__name__

    act_name = type(value).__name__
    if not isinstance(value, tuple(correction)):
        raise TypeError(f"'{name}' must be type {exp_names}, got {act_name}.")


def _check_inner_type(
    name: str,
    value: Sequence[Any],
    expected: Type | tuple[Type],
) -> None:
    """
    Check the type of items within a sequence against an expected type.

    Note that this function only checks the types within the sequence, not the
    type of the sequence itself. If it is necessary to check the "outer" type of
    the sequence, use `_check_type`.

    Parameters
    ----------
    name : str
        Name of the parameter being checked, used in error messages.
    value : Sequence[Any]
        Sequence of values to be checked.
    expected : Type or tuple[Type]
        Expected type(s) for each item in the sequence.

    Raises
    ------
    TypeError
        If any object(s) within the sequence don't match the expected type.

    """
    if isinstance(expected, tuple):
        exp_names = "(" + ", ".join([t.__name__ for t in expected]) + ")"
    else:
        exp_names = expected.__name__
        expected = (expected,)

    if not all(isinstance(item, expected) for item in value):
        raise TypeError(f"All items in '{name}' must be type {exp_names}.")


def _check_only_one(conditions: Sequence[bool], message: str) -> None:
    """
    Check that exactly one condition in a sequence of conditions is True.

    Parameters
    ----------
    conditions : Sequence[bool]
        A sequence of boolean conditions to check.
    message : str
        Error message to raise if not exactly one condition is True.

    Raises
    ------
    ValueError
        If not exactly one condition is True. A custom error message is provided
        to clarify the expected conditions.

    """
    if sum(conditions) != 1:
        raise ValueError(message)
