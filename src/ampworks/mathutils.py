"""
General-purpose mathematical utilities for array and numerical computations.
Provides reusable functions to simplify common tasks in data analysis and math.

"""
from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

__all__ = [
    'combinations',
    'aggregate_over_x',
]

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset


def aggregate_over_x(
    datasets: Sequence[Dataset],
    x: str,
    y: str,
    n: int = 100,
) -> Dataset:
    """
    Aggregate datasets over a shared `x` grid.

    The function finds the overlapping range of `x` across all datasets,
    interpolates each dataset's `y` onto an evenly spaced `x` grid, and then
    computes summary statistics for `y` at each grid point.

    Parameters
    ----------
    datasets : Sequence[Dataset]
        Datasets to aggregate. Each dataset must contain requested x/y columns.
    x : str
        Column name used to build the interpolation grid.
    y : str
        Column name interpolated and aggregated on the shared grid.
    n : int, optional
        Number of evenly spaced points in the shared grid. Default is 100.

    Returns
    -------
    data : Dataset
        Dataset with columns for the `x` grid and aggregated `y` statistics,
        including: mean, standard deviation, minimum, and maximum.

    Raises
    ------
    TypeError
        If any input argument has an invalid type.
    ValueError
        If `datasets` is empty, if `n < 2`, if the requested columns are missing
        from any dataset, or if the datasets have no overlapping range in `x`.

    Examples
    --------
    The code snippet below demonstrates how to use `aggregate_over_x`. Here, we
    load a beginning-of-life and end-of-life cell dataset from the `datasets`
    subpackage. Combining these datasets has no particular physical meaning, but
    serves to illustrate the function.

    .. code-block:: python

        import ampworks as amp
        import matplotlib.pyplot as plt

        data1, data2 = amp.datasets.load_datasets(
            'dqdv/cell1_rough', 'dqdv/cell2_rough',
        )

        avg = amp.mathutils.aggregate_over_x([data1, data2], 'Volts', 'Ah')
        dwn = avg.downsample(n=25)

        errbar = plt.errorbar(
            dwn['Ah_mean'], dwn['Volts'], xerr=dwn['Ah_std'], fmt='.',
        )
        fill_x = plt.fill_betweenx(
            dwn['Volts'], dwn['Ah_min'], dwn['Ah_max'], alpha=0.2,
        )

        plt.legend([errbar, fill_x], ["Mean +/- Std", "Min-Max Range"])
        plt.xlabel("Discharge Capacity [Ah]")
        plt.ylabel("Voltage [V]")

        plt.show()

    """
    from ampworks import Dataset
    from ampworks._checks import _check_columns, _check_type

    _check_type('datasets', datasets, Sequence)
    _check_type('x', x, str)
    _check_type('y', y, str)
    _check_type('n', n, int)

    if len(datasets) == 0:
        raise ValueError("'datasets' must contain at least one dataset.")
    if n < 2:
        raise ValueError("'n' must be at least 2.")

    for i, data in enumerate(datasets):
        _check_type(f"datasets[{i}]", data, (Dataset, pd.DataFrame))
        _check_columns(data, [x, y])

    lo = max(data[x].min() for data in datasets)
    hi = min(data[x].max() for data in datasets)
    if lo >= hi:
        raise ValueError(f"No overlapping range found for x='{x}'.")

    x_grid = np.linspace(lo, hi, n)

    interpolated = np.empty((len(datasets), n))
    for i, data in enumerate(datasets):
        x_vals = data[x].to_numpy()
        y_vals = data[y].to_numpy()

        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        y_vals = y_vals[order]

        # Keep first occurrence of duplicate x values for stable interpolation
        uniq_x, uniq_idx = np.unique(x_vals, return_index=True)
        uniq_y = y_vals[uniq_idx]

        interpolated[i] = np.interp(x_grid, uniq_x, uniq_y)

    zero_std = np.zeros(n)
    use_std = len(datasets) > 1

    return Dataset({
        x: x_grid,
        f"{y}_mean": interpolated.mean(axis=0),
        f"{y}_std": interpolated.std(axis=0, ddof=1) if use_std else zero_std,
        f"{y}_min": interpolated.min(axis=0),
        f"{y}_max": interpolated.max(axis=0),
    })


def combinations(
    values: Sequence[np.ndarray],
    names: Sequence[str] = None,
) -> list[dict]:
    """
    Generate all value combinations.

    Parameters
    ----------
    values : Sequence[1D array]
        Variable values. Array `i` corresponds to `names[i]`, if provided.
    names : Sequence[str], optional
        Variable names. Defaults to `range(N)` when not provided, where `N`
        is the length of 'values', i.e., how many arrays are in the sequence.

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
