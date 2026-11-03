from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ampworks import _checks as _chk
from ampworks.columns._backend import (
    _ah_wh,
    _ah_wh_cumulative,
    _ah_wh_throughput,
)

if TYPE_CHECKING:
    from ampworks import Dataset


def add_energy(
    data: Dataset,
    *,
    by: str = 'State',
    output_col: str = 'Wh',
    seconds_col: str = 'Seconds',
    watts_col: str = 'Watts',
) -> Dataset:
    """
    Add an energy column to a dataset.

    Calculates energy as the integral of absolute power over time, meaning
    all outputs are non-negative. Resets to zero each time a change in value
    occurs in the `by` column. Integration is performed using the trapezoidal
    rule.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    by : str, optional
        The column used to define groups where energy resets to zero. Defaults
        to 'State', which resets to zero on state changes.
    output_col : str, optional
        Name of the new energy column to add, by default 'Wh'.
    seconds_col : str, optional
        Name of column containing time in seconds, by default 'Seconds'.
    watts_col : str, optional
        Name of column containing power in watts, by default 'Watts'.

    Returns
    -------
    Dataset
        A modified copy of the input data, with an energy column.

    See Also
    --------
    ~ampworks.columns.add_power :
        Add a power column, based on supplied current and voltage columns.

    Notes
    -----
    The trapezoidal rule is used for integration, but this only approximates the
    true energy. Results are more accurate with higher time resolution, but
    can be inaccurate for low time resolution. In worst cases, data may include
    switches from an active step (charge or discharge) to a rest step without
    any time records during the rest. The trapezoidal rule then assumes a linear
    change in power from the last active step to the next active step, even
    though the true power was zero during the rest, leading to significant
    errors. Be aware of this limitation when using low time resolution data.

    Examples
    --------
    Below we add an energy column to a dataset using default settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_energy(data)

    You may have cases where you want energy to reset on step or cycle changes
    instead of state changes. In these cases, use a different `by` column to
    control where energy resets occur.

    >>> ds = add_energy(data, by='Step')
    >>> ds = add_energy(data, by='Cycle')

    """
    ds = data.copy()

    ds[output_col] = _ah_wh(
        data=ds,
        by=by,
        seconds_col=seconds_col,
        value_col=watts_col,
    )

    return ds


def add_cumulative_energy(
    data: Dataset,
    *,
    method: Literal['integral', 'column'] = 'integral',
    output_col: str = 'CumulativeWh',
    seconds_col: str = 'Seconds',
    watts_col: str = 'Watts',
    state_col: str = 'State',
    wh_col: str = 'Wh',
) -> Dataset:
    """
    Add a cumulative energy column to a dataset.

    Calculates cumulative energy assuming increasing energy on charge and
    decreasing energy on discharge. Differs from the `add_energy` function
    in that cumulative energy never resets to zero, but instead accumulates
    over the entire dataset. Additionally, since the accumulation is based on
    the sign of power, cumulative energy can be negative, positive, or zero.

    How charging and discharging segments are defined differs depending on the
    method, using signs of power (`method='integral'`) or values of {'C', 'D',
    or 'R'} in the state column (`method='column'`). The standardized sign
    convention for power across the package is positive and negative on charge
    and discharge, respectively, and exactly zero for rests. If your data uses a
    different sign convention, it must be corrected before using this function.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column'], optional
        The method to use for calculating the cumulative energy, by default
        'integral'. See notes for info on method differences and assumptions.
    output_col : str, optional
        Name of cumulative energy column to add; defaults to 'CumulativeWh'.
    seconds_col : str, optional
        Name of column containing time in seconds, by default 'Seconds'.
        Only used when `method='integral'`.
    watts_col : str, optional
        Name of column containing power in watts, by default 'Watts'.
        Only used when `method='integral'`.
    state_col : str, optional
        Name of column containing the state, by default 'State'. Only
        used when `method='column'`.
    wh_col : str, optional
        Name of column containing energy in watt-hours, by default 'Wh'.
        Only used when `method='column'`.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a cumulative energy column.

    See Also
    --------
    ~ampworks.columns.add_power :
        Add a power column, based on supplied current and voltage columns.
    ~ampworks.columns.add_energy :
        Add an energy column, with resets to zero at the start of each step,
        compatible with the `'column'` method of this function.
    ~ampworks.columns.add_state :
        Add a state column to a dataset, compatible with the `'column'`
        method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate power over
    time, but this only approximates the true cumulative energy. Results are
    more accurate with higher time resolution, but can be inaccurate for low
    time resolution.

    The `'column'` method assumes that input data already contains a column
    with watt-hour values. It assumes that all Wh values are non-negative, and
    that any non-monotonic behavior (i.e., decreases from one row to another)
    are only due to resets back to zero. Then, cumulative energy is calculated
    by accumulating differences between rows, ignoring resets. The direction of
    accumulation (increasing or decreasing) relies on the `state_col` column,
    increasing when the state is 'C' (charge) and decreasing when the state is
    'D' (discharge). This requires a column to be present, with definitions that
    'C' is charge, 'D' is discharge, and 'R' is rest.

    Examples
    --------
    Below we add a cumulative energy column to a dataset using the default
    settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_cumulative_energy(data)

    If your data already contains a column with watt-hour values, you can use
    the `'column'` method. However, this assumes the existing energy column
    is correct and always positive, and that a state column is present with the
    definitions that 'C' is charge, 'D' is discharge, and 'R' is rest. These
    columns can be added using the `add_state` and `add_energy` functions, if
    they are missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_state(data)
    >>> ds = add_energy(ds)
    >>> ds = add_cumulative_energy(ds, method='column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    ds = data.copy()

    ds[output_col] = _ah_wh_cumulative(
        data=ds,
        method=method,
        seconds_col=seconds_col,
        value_col=watts_col,
        state_col=state_col,
        valueh_col=wh_col,
    )

    return ds


def add_throughput_energy(
    data: Dataset,
    *,
    method: Literal['integral', 'column'] = 'integral',
    output_col: str = 'ThroughputWh',
    seconds_col: str = 'Seconds',
    watts_col: str = 'Watts',
    wh_col: str = 'Wh',
) -> Dataset:
    """
    Add a throughput energy column to a dataset.

    Calculates throughput energy using either the absolute value of power or
    values from a watt-hour column. Differs from the other energy functions,
    `add_energy` and `add_cumulative_energy`, in that throughput energy always
    increases, regardless of the sign on power, and there are no resets to zero.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column'], optional
        The method to calculate throughput energy, by default 'integral'. See
        notes for info on method differences and assumptions.
    output_col : str, optional
        Name of throughput energy column to add; defaults to 'ThroughputWh'.
    seconds_col : str, optional
        Name of column containing time in seconds, by default 'Seconds'.
        Only used when `method='integral'`.
    watts_col : str, optional
        Name of column containing power in watts, by default 'Watts'.
        Only used when `method='integral'`.
    wh_col : str, optional
        Name of column containing energy in watt-hours, by default 'Wh'.
        Only used when `method='column'`.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a throughput energy column.

    See Also
    --------
    ~ampworks.columns.add_power :
        Add a power column, based on supplied current and voltage columns.
    ~ampworks.columns.add_energy :
        Add an energy column, with resets to zero at the start of each step,
        compatible with the `'column'` method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate the absolute
    value of power over time, but this only approximates the true throughput
    energy. Results are more accurate with higher time resolution, but can be
    inaccurate for low time resolution.

    The `'column'` method assumes that input data already contains a column
    with watt-hour values. It assumes that all Wh values are non-negative, and
    that any non-monotonic behavior (i.e., decreases from one row to another)
    are only due to resets back to zero. Then, throughput energy is calculated
    by accumulating the absolute differences between rows, ignoring resets.

    Examples
    --------
    Below we add a throughput energy column to a dataset using the default
    settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_throughput_energy(data)

    If your data already contains a column with watt-hour values, you can use
    the `'column'` method. However, this assumes the existing energy column
    is correct and always positive. This column can be added using `add_energy`,
    if it is missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_energy(data)
    >>> ds = add_throughput_energy(ds, method='column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    ds = data.copy()

    ds[output_col] = _ah_wh_throughput(
        data=ds,
        method=method,
        seconds_col=seconds_col,
        value_col=watts_col,
        valueh_col=wh_col,
    )

    return ds
