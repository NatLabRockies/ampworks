from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from ampworks import _checks as _chk
from ampworks._core._std_names import SECONDS, STATE, WH, WATTS
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
    which: str = STATE,
    col_name: str = WH,
    seconds_alias: str = SECONDS,
    watts_alias: str = WATTS,
) -> Dataset:
    """
    Add an energy column to a dataset.

    Calculates energy as the integral of absolute power over time, meaning
    all outputs are non-negative. Resets to zero each time a change in value
    occurs in the column specified by the `which` argument. Integration is
    performed using the trapezoidal rule.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to define groups where energy resets to zero. Defaults
        to state-based, which resets to zero on state changes.
    col_name : str, optional
        Name of the new energy column to add; defaults to standard name.
    seconds_alias : str, optional
        Name of column containing time in seconds; defaults to standard name.
    watts_alias : str, optional
        Name of column containing power in watts; defaults to standard name.

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
    instead of state changes. In these cases, use a different `which` column to
    control where energy resets occur.

    >>> ds = add_energy(data, which='Step')
    >>> ds = add_energy(data, which='Cycle')

    """
    ds = data.copy()

    ds[col_name] = _ah_wh(
        data=ds,
        which=which,
        seconds_alias=seconds_alias,
        value_alias=watts_alias,
    )

    return ds


def add_cumulative_energy(
    data: Dataset,
    *,
    method: Literal['integral', 'wh_column'] = 'integral',
    col_name: str = 'CumulativeWh',
    seconds_alias: str = SECONDS,
    watts_alias: str = WATTS,
    state_alias: str = STATE,
    wh_alias: str = WH,
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
    or 'R'} in the state column (`method='wh_column'`). The standardized sign
    convention for power across the package is positive and negative on charge
    and discharge, respectively, and exactly zero for rests. If your data uses a
    different sign convention, it must be corrected before using this function.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'wh_column'], optional
        The method to use for calculating the cumulative energy, by default
        'integral'. See notes for info on method differences and assumptions.
    col_name : str, optional
        Name of cumulative energy column to add; defaults to 'CumulativeWh'.
    seconds_alias : str, optional
        Name of column containing time in seconds; defaults to standard name.
        Only used when `method='integral'`.
    watts_alias : str, optional
        Name of column containing power in watts; defaults to standard name.
        Only used when `method='integral'`.
    state_alias : str, optional
        Name of column containing the state; defaults to standard name. Only
        used when `method='wh_column'`.
    wh_alias : str, optional
        Name of column containing energy in watt-hours; defaults to standard
        name. Only used when `method='wh_column'`.

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
        compatible with the `'wh_column'` method of this function.
    ~ampworks.columns.add_state :
        Add a state column to a dataset, compatible with the `'wh_column'`
        method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate power over
    time, but this only approximates the true cumulative energy. Results are
    more accurate with higher time resolution, but can be inaccurate for low
    time resolution.

    The `'wh_column'` method assumes that input data already contains a column
    with watt-hour values. It assumes that all Wh values are non-negative, and
    that any non-monotonic behavior (i.e., decreases from one row to another)
    are only due to resets back to zero. Then, cumulative energy is calculated
    by accumulating differences between rows, ignoring resets. The direction of
    accumulation (increasing or decreasing) relies on the `state_alias` column,
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
    the `'wh_column'` method. However, this assumes the existing energy column
    is correct and always positive, and that a state column is present with the
    definitions that 'C' is charge, 'D' is discharge, and 'R' is rest. These
    columns can be added using the `add_state` and `add_energy` functions, if
    they are missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_state(data)
    >>> ds = add_energy(ds)
    >>> ds = add_cumulative_energy(ds, method='wh_column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'wh_column'})

    ds = data.copy()

    ds[col_name] = _ah_wh_cumulative(
        data=ds,
        method=method.split('_')[-1],  # compatible with generic integral/column
        seconds_alias=seconds_alias,
        value_alias=watts_alias,
        state_alias=state_alias,
        valueh_alias=wh_alias,
    )

    return ds


def add_throughput_energy(
    data: Dataset,
    *,
    method: Literal['integral', 'wh_column'] = 'integral',
    col_name: str = 'ThroughputWh',
    seconds_alias: str = SECONDS,
    watts_alias: str = WATTS,
    wh_alias: str = WH,
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
    method : Literal['integral', 'wh_column'], optional
        The method to calculate throughput energy, by default 'integral'. See
        notes for info on method differences and assumptions.
    col_name : str, optional
        Name of throughput energy column to add; defaults to 'ThroughputWh'.
    seconds_alias : str, optional
        Name of column containing time in seconds; defaults to standard name.
        Only used when `method='integral'`.
    watts_alias : str, optional
        Name of column containing power in watts; defaults to standard name.
        Only used when `method='integral'`.
    wh_alias : str, optional
        Name of column containing energy in watt-hours; defaults to standard
        name. Only used when `method='wh_column'`.

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
        compatible with the `'wh_column'` method of this function.

    Notes
    -----
    The `'integral'` method uses the trapezoidal rule to integrate the absolute
    value of power over time, but this only approximates the true throughput
    energy. Results are more accurate with higher time resolution, but can be
    inaccurate for low time resolution.

    The `'wh_column'` method assumes that input data already contains a column
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
    the `'wh_column'` method. However, this assumes the existing energy column
    is correct and always positive. This column can be added using `add_energy`,
    if it is missing, as demonstrated below.

    >>> data = amp.Dataset(...)
    >>> ds = add_energy(data)
    >>> ds = add_throughput_energy(ds, method='wh_column')

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'wh_column'})

    ds = data.copy()

    ds[col_name] = _ah_wh_throughput(
        data=ds,
        method=method.split('_')[-1],  # compatible with generic integral/column
        seconds_alias=seconds_alias,
        value_alias=watts_alias,
        valueh_alias=wh_alias,
    )

    return ds
