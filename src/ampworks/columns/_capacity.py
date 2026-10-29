from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ampworks import _checks as _chk
from ampworks._core._std_names import SECONDS, AMPS, STATE, AH
from ampworks.columns._backend import (
    _ah_wh,
    _ah_wh_cumulative,
    _ah_wh_throughput,
)

if TYPE_CHECKING:
    from ampworks import Dataset


def add_capacity(
    data: Dataset,
    *,
    by: str = STATE,
    output_col: str = AH,
    seconds_col: str = SECONDS,
    amps_col: str = AMPS,
) -> Dataset:
    """
    Add a capacity column to a dataset.

    Calculates capacity as the integral of absolute current over time, meaning
    all outputs are non-negative. Resets to zero each time a change in value
    occurs in the `by` column. Integration is performed using the trapezoidal
    rule.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    by : str, optional
        The column used to define groups where capacity resets to zero. Defaults
        to state-based, which resets zero on state changes.
    output_col : str, optional
        Name of the new capacity column to add; defaults to standard name.
    seconds_col : str, optional
        Name of column containing time in seconds; defaults to standard name.
    amps_col : str, optional
        Name of column containing current in amps; defaults to standard name.

    Returns
    -------
    Dataset
        A modified copy of the input data, with a capacity column.

    Notes
    -----
    The trapezoidal rule is used for integration, but this only approximates the
    true capacity. Results are more accurate with higher time resolution, but
    can be inaccurate for low time resolution. In worst cases, data may include
    switches from an active step (charge or discharge) to a rest step without
    any time records during the rest. The trapezoidal rule then assumes a linear
    change in current from the last active step to the next active step, even
    though the true current was zero during the rest, leading to significant
    errors. Be aware of this limitation when using low time resolution data.

    Examples
    --------
    Below we add a capacity column to a dataset using default settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_capacity(data)

    You may have cases where you want capacity to reset on step or cycle changes
    instead of state changes. In these cases, use a different `by` column to
    control where capacity resets occur.

    >>> ds = add_capacity(data, by='Step')
    >>> ds = add_capacity(data, by='Cycle')

    """
    ds = data.copy()

    ds[output_col] = _ah_wh(
        data=ds,
        by=by,
        seconds_col=seconds_col,
        value_col=amps_col,
    )

    return ds


def add_cumulative_capacity(
    data: Dataset,
    *,
    method: Literal['integral', 'column'] = 'integral',
    output_col: str = 'CumulativeAh',
    seconds_col: str = SECONDS,
    amps_col: str = AMPS,
    state_col: str = STATE,
    ah_col: str = AH,
) -> Dataset:
    """
    Add a cumulative capacity column to a dataset.

    Calculates cumulative capacity assuming increasing capacity on charge and
    decreasing capacity on discharge. Differs from the `add_capacity` function
    in that cumulative capacity never resets to zero, but instead accumulates
    over the entire dataset. Additionally, since the accumulation is based on
    the sign of current, cumulative capacity can be negative, positive, or zero.

    How charging and discharging segments are defined differs depending on the
    method, using signs of current (`method='integral'`) or values of {'C', 'D',
    or 'R'} in the state column (`method='column'`). The standardized sign
    convention for current across the package is positive and negative on charge
    and discharge, respectively, and exactly zero for rests. If your data uses a
    different sign convention, it must be corrected before using this function.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column'], optional
        The method to use for calculating the cumulative capacity, by default
        'integral'. See notes for info on method differences and assumptions.
    output_col : str, optional
        Name of cumulative capacity column to add; defaults to 'CumulativeAh'.
    seconds_col : str, optional
        Name of column containing time in seconds; defaults to standard name.
        Only used when `method='integral'`.
    amps_col : str, optional
        Name of column containing current in amps; defaults to standard name.
        Only used when `method='integral'`.
    state_col : str, optional
        Name of column containing the state; defaults to standard name. Only
        used when `method='column'`.
    ah_col : str, optional
        Name of column containing capacity in ampere-hours; defaults to standard
        name. Only used when `method='column'`.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a cumulative capacity column.

    See Also
    --------
    ~ampworks.columns.add_capacity :
        Add a capacity column, with resets to zero at the start of each step,
        compatible with the `'column'` method of this function.
    ~ampworks.columns.add_state :
        Add a state column to a dataset, compatible with the `'column'`
        method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate current over
    time, but this only approximates the true cumulative capacity. Results are
    more accurate with higher time resolution, but can be inaccurate for low
    time resolution.

    The `'column'` method assumes that input data already contains a column
    with ampere-hour values. It assumes that all Ah values are non-negative, and
    that any non-monotonic behavior (i.e., decreases from one row to another)
    are only due to resets back to zero. Then, cumulative capacity is calculated
    by accumulating differences between rows, ignoring resets. The direction of
    accumulation (increasing or decreasing) relies on the `state_col` column,
    increasing when the state is 'C' (charge) and decreasing when the state is
    'D' (discharge). This requires a column to be present, with definitions that
    'C' is charge, 'D' is discharge, and 'R' is rest.

    Examples
    --------
    Below we add a cumulative capacity column to a dataset using the default
    settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_cumulative_capacity(data)

    If your data already contains a column with ampere-hour values, you can use
    the `'column'` method. However, this assumes the existing capacity column
    is correct and always positive, and that a state column is present with the
    definitions that 'C' is charge, 'D' is discharge, and 'R' is rest. These
    columns can be added using the `add_state` and `add_capacity` functions, if
    they are missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_state(data)
    >>> ds = add_capacity(ds)
    >>> ds = add_cumulative_capacity(ds, method='column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    ds = data.copy()

    ds[output_col] = _ah_wh_cumulative(
        data=ds,
        method=method,
        seconds_col=seconds_col,
        value_col=amps_col,
        state_col=state_col,
        valueh_col=ah_col,
    )

    return ds


def add_throughput_capacity(
    data: Dataset,
    *,
    method: Literal['integral', 'column'] = 'integral',
    output_col: str = 'ThroughputAh',
    seconds_col: str = SECONDS,
    amps_col: str = AMPS,
    ah_col: str = AH,
) -> Dataset:
    """
    Add a throughput capacity column to a dataset.

    Calculates throughput capacity using either the absolute value of current or
    values from an ampere-hour column. Differs from other capacity functions,
    `add_capacity` and `add_cumulative_capacity`, in that throughput capacity
    always increases, regardless of the sign on current, and there are no resets
    to zero.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column'], optional
        The method to calculate throughput capacity, by default 'integral'. See
        notes for info on method differences and assumptions.
    output_col : str, optional
        Name of throughput capacity column to add; defaults to 'ThroughputAh'.
    seconds_col : str, optional
        Name of column containing time in seconds; defaults to standard name.
        Only used when `method='integral'`.
    amps_col : str, optional
        Name of column containing current in amps; defaults to standard name.
        Only used when `method='integral'`.
    ah_col : str, optional
        Name of column containing capacity in ampere-hours; defaults to standard
        name. Only used when `method='column'`.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a throughput capacity column.

    See Also
    --------
    ~ampworks.columns.add_capacity :
        Add a capacity column, with resets to zero at the start of each step,
        compatible with the `'column'` method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate the absolute
    value of current over time, but this only approximates the true throughput
    capacity. Results are more accurate with higher time resolution, but can be
    inaccurate for low time resolution.

    The `'column'` method assumes that input data already contains a column
    with ampere-hour values. It assumes that all Ah values are non-negative, and
    that any non-monotonic behavior (i.e., decreases from one row to another)
    are only due to resets back to zero. Then, throughput capacity is calculated
    by accumulating the absolute differences between rows, ignoring resets.

    Examples
    --------
    Below we add a throughput capacity column to a dataset using the default
    settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_throughput_capacity(data)

    If your data already contains a column with ampere-hour values, you can use
    the `'column'` method. However, this assumes the existing capacity column
    is correct and always positive. Add a compatible capacity column using the
    `add_capacity`, if it is missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_capacity(data)
    >>> ds = add_throughput_capacity(ds, method='column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    ds = data.copy()

    ds[output_col] = _ah_wh_throughput(
        data=ds,
        method=method,
        seconds_col=seconds_col,
        value_col=amps_col,
        valueh_col=ah_col,
    )

    return ds


def add_equivalent_full_cycles(
    data: Dataset,
    *,
    nominal_ah: float,
    output_col: str = 'EFC',
    throughput_col: str = 'ThroughputAh',
) -> Dataset:
    """
    Add an equivalent full cycles column to a dataset.

    Calculates equivalent full cycles as the throughput capacity divided by
    twice the nominal capacity. This is a common metric for battery cycling, and
    is used to quantify the number of full charge-discharge cycles a battery has
    undergone, even if the battery has only been partially cycled. The added
    column is always non-negative, never decreases, and is continuous, rather
    than only incrementing in discrete steps.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    nominal_ah : float
        The nominal capacity of the battery in ampere-hours.
    output_col : str, optional
        Name of the column to add, by default 'EFC'.
    throughput_col : str, optional
        Name of the column containing throughput capacity in ampere-hours, by
        default 'ThroughputAh'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with an added column for equivalent
        full cycles.

    Raises
    ------
    ValueError
        If `nominal_ah` is zero.

    See Also
    --------
    ~ampworks.columns.add_throughput_capacity :
        Add a throughput capacity column, which is used to calculate equivalent
        full cycles.

    Examples
    --------
    Below we add an equivalent full cycles column to a dataset using the default
    settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_equivalent_full_cycles(data, nominal_ah=2.0)

    If the throughput capacity column is missing, it can be added using the
    `add_throughput_capacity` function, before computing the equivalent full
    cycles, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_throughput_capacity(data)
    >>> ds = add_equivalent_full_cycles(ds, nominal_ah=2.0)

    """
    _chk._check_columns(data, [throughput_col])

    if nominal_ah == 0:
        raise ValueError("'nominal_ah' must be nonzero.")

    ds = data.copy()
    ds[output_col] = ds[throughput_col] / (2.0 * abs(nominal_ah))
    return ds
