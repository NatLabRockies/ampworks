from __future__ import annotations

from warnings import warn
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from scipy.integrate import cumulative_trapezoid

from ampworks import _checks as _chk

if TYPE_CHECKING:
    from pandas import Series
    from ampworks import Dataset

__all__ = [
    '_instance_nums',
    '_ah_wh',
    '_ah_wh_cumulative',
    '_ah_wh_throughput',
]


def _instance_nums(
    data: Dataset,
    *,
    which: str,
    cycle_alias: str | None,
    cycle_resets: bool,
    fast: bool,
) -> Series:
    """
    Helper function for instance numbers. Not part of the public API. Returns a
    Series instead of a Dataset, which can still be used for creating groups or
    later assigned as as new column.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str
        The column used to detect and group instances.
    cycle_alias : str | None
        The column containing cycle numbers. Only used if `cycle_resets=True`.
    cycle_resets : bool
        Whether the instance numbers reset at the start of each cycle. If True,
        instance numbers are unique within each cycle, not globally.
    fast : bool
        If True, returns raw instance numbers. See notes section for details.

    Returns
    -------
    instance_nums : Series
        Instance numbers for each row in the dataset, using 1-based indexing.

    Raises
    ------
    ValueError
        If `cycle_resets` is True and `cycle_alias` is None.

    Warnings
    --------
    UserWarning
        Cycle resets are ignored when `fast=True`, even if requested.

    Notes
    -----
    With `fast=True`, cycle resets are ignored and the result is just a
    monotonically increasing changeover counter, not a per-group occurrence
    count.

    """
    first = data[which].iloc[0]
    changeovers = (data[which] != data[which].shift(fill_value=first))

    if cycle_resets and fast:
        warn("Cycle resets are ignored when 'fast=True'.", UserWarning)

    raw = changeovers.astype(bool).cumsum() + 1
    if fast:
        return raw

    if cycle_resets and (cycle_alias is None):
        raise ValueError("'cycle_alias' is required when cycle_resets=True.")

    grouping = [which, cycle_alias] if cycle_resets else [which]

    instance_nums = (
        data
        .assign(_raw=raw)
        .groupby(grouping)['_raw']
        .rank(method='dense')
        .astype(int)
    )

    return instance_nums


def _ah_wh(
    data: Dataset,
    *,
    which: str,
    seconds_alias: str,
    value_alias: str,
) -> Series:
    """
    Helper function for `add_capacity` and `add_energy`.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str
        The column used to define groups where Ah or Wh resets to zero.
    seconds_alias : str
        Name of the column containing time in seconds.
    value_alias : str
        Name of the column containing current in amps or power in watts.

    Returns
    -------
    ahwh : Series
        Non-negative Ah or Wh values for each row, integrated from the absolute
        value of current or power and reset to zero at the start of each group.

    See Also
    --------
    _ah_wh_cumulative : Signed, non-resetting cumulative Ah or Wh.
    _ah_wh_throughput : Absolute, non-resetting throughput Ah or Wh.

    Notes
    -----
    Values are always non-negative so that the result stays compatible with
    the `'column'` method of `_ah_wh_cumulative`/`_ah_wh_throughput`, which
    requires a non-negative Ah/Wh column that only decreases at resets.

    """
    _chk._check_columns(data, [which, seconds_alias, value_alias])

    instance_nums = _instance_nums(
        data=data,
        which=which,
        cycle_alias=None,
        cycle_resets=False,
        fast=True,
    )

    # integrate once over full series, then re-reference groups to their starts
    x = data[seconds_alias] / 3600  # seconds to hours
    y = data[value_alias].abs()

    cumulative = pd.Series(
        cumulative_trapezoid(y=y, x=x, initial=0),
        index=data.index,
    )
    offsets = (
        cumulative
        .groupby([data[which], instance_nums])
        .transform('first')
    )

    ahwh = cumulative - offsets

    return ahwh


def _ah_wh_cumulative(
    data: Dataset,
    *,
    method: Literal['integral', 'column'],
    seconds_alias: str,
    value_alias: str,
    state_alias: str,
    valueh_alias: str,
) -> Series:
    """
    Helper function for `add_cumulative_capacity` and `add_cumulative_energy`.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column']
        The method to use for calculating the cumulative quantity.
    seconds_alias : str
        Name of the column containing time in seconds.
    value_alias : str
        Name of the column containing current in amps or power in watts.
    state_alias : str
        Name of the column containing the state, with values of {'C', 'D', 'R'}.
    valueh_alias : str
        Name of the column containing ampere-hour or watt-hour values. Only
        used for `method='column'`, which requires non-negative values that
        only decrease at resets (e.g., as produced by `_ah_wh`).

    Returns
    -------
    ahwh : Series
        Cumulative Ah or Wh values for each row in the dataset.

    See Also
    --------
    _ah_wh : Produces a compatible non-negative, reset-at-group-start column.

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    # required columns depends on method
    use_ahwh = (method == 'column')
    if use_ahwh:
        required = {valueh_alias, state_alias}
    else:
        required = {seconds_alias, value_alias}

    _chk._check_columns(data, required)

    if method == 'integral':
        x = data[seconds_alias] / 3600  # seconds to hours
        y = data[value_alias]

        ahwh = cumulative_trapezoid(y=y, x=x, initial=0)

    else:
        # ah/wh_column methods assume all >= 0 values for capacity/energy, that
        # non-monotonic behavior is only due to resets, and that state column
        # is present with 'C' for charge, 'D' for discharge, and 'R' for rest

        if not (data[valueh_alias] >= 0).all():
            raise ValueError(
                f"All values in column '{valueh_alias}' must be non-negative."
            )

        valid = {'C', 'D', 'R'}
        if not data[state_alias].isin(valid).all():
            raise ValueError(
                f"All values in column '{state_alias}' must be one of {valid}."
            )

        increments = data[valueh_alias].diff().fillna(0)
        signs = data[state_alias].map({'C': 1, 'D': -1, 'R': 0})

        # clip lower bound to 0 since this only occurs when zeroing out the
        # column at new steps, and we don't want to subtract on these resets
        signed_increments = signs * np.clip(increments, 0, None)
        ahwh = signed_increments.cumsum()

    return pd.Series(ahwh)


def _ah_wh_throughput(
    data: Dataset,
    *,
    method: Literal['integral', 'column'],
    seconds_alias: str,
    value_alias: str,
    valueh_alias: str,
) -> Series:
    """
    Helper function for `add_throughput_capacity` and `add_throughput_energy`.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    method : Literal['integral', 'column']
        The method to use for calculating the throughput quantity.
    seconds_alias : str
        Name of the column containing time in seconds.
    value_alias : str
        Name of the column containing current in amps or power in watts.
    valueh_alias : str
        Name of the column containing ampere-hour or watt-hour values. Only
        used for `method='column'`, which requires non-negative values that
        only decrease at resets (e.g., as produced by `_ah_wh`).

    Returns
    -------
    ahwh : Series
        Throughput Ah or Wh values for each row in the dataset.

    See Also
    --------
    _ah_wh : Produces a compatible non-negative, reset-at-group-start column.

    """
    method = method.lower()

    _chk._check_literal('method', method, {'integral', 'column'})

    # required columns depends on method
    use_ahwh = (method == 'column')
    if use_ahwh:
        required = {valueh_alias}
    else:
        required = {seconds_alias, value_alias}

    _chk._check_columns(data, required)

    if method == 'integral':
        x = data[seconds_alias] / 3600  # seconds to hours
        y = data[value_alias]

        ahwh = cumulative_trapezoid(y=y.abs(), x=x, initial=0)  # use abs()

    else:
        # ah/wh_column methods assume all >= 0 values for capacity/energy, and
        # that non-monotonic behavior is only due to resets

        if not (data[valueh_alias] >= 0).all():
            raise ValueError(
                f"All values in column '{valueh_alias}' must be non-negative."
            )

        y = data[valueh_alias]

        was_reset = (y.diff() < 0)
        value_before_reset = y.shift(1)
        reset_amounts = value_before_reset.where(was_reset, 0.0)

        ahwh = y + reset_amounts.cumsum()

    return pd.Series(ahwh)
