"""
General-purpose mathematical utilities for array and numerical computations.
Provides reusable functions to simplify common tasks in data analysis and math.

"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from numpy import ndarray


def combinations(values: list[ndarray], names: list[str] = None) -> list[dict]:
    """
    Generate all value combinations.

    Parameters
    ----------
    values : list[1D array]
        Variable values. Array `i` corresponds to `names[i]`, if provided.
    names : list[str], optional
        Variable names. Defaults to `range(N)` when not provided, where `N`
        is the length of 'values', i.e., how many arrays are in the list.

    Returns
    -------
    combinations : list[dict]
        Dictionaries for each possible combination of values.

    """
    import itertools

    if names is None:
        names = [i for i in range(len(values))]

    combinations = []
    for combination in itertools.product(*values):
        combinations.append({k: v for k, v in zip(names, combination)})

    return combinations
