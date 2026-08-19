from __future__ import annotations

from typing import TYPE_CHECKING

from ampworks import _checks as _chk
from ampworks.columns._backend import _instance_nums

if TYPE_CHECKING:
    from ampworks import Dataset


def add_instance_nums(
    data: Dataset,
    *,
    which: str = 'Step',
    col_name: str = 'InstanceNum',
    cycle_alias: str = 'Cycle',
    cycle_resets: bool = True,
    fast: bool = False,
) -> Dataset:
    """
    Add an instance numbers column to a dataset.

    Instance numbers uniquely identify repeated occurrences of a group. For
    example, if a single HPPC cycle contains ten discharge pulses that all
    share the same step number (e.g., 5), instance numbers label them 1
    through 10. Numbering can reset every cycle or stay globally unique,
    depending on `cycle_resets`. Numbering is 1-based, so the first instance
    of a group is always 1, not 0.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to detect and group instances, by default 'Step'.
    col_name : str, optional
        Name of the column to add, by default 'InstanceNum'.
    cycle_alias : str, optional
        Name of the column containing cycle numbers, by default 'Cycle'. Only
        used if the `cycle_resets` argument is True.
    cycle_resets : bool, optional
        Whether the instance numbers reset at the start of each cycle. If True
        (default), instance numbers are unique within each cycle, not globally.
    fast : bool, optional
        If True, returns the raw instance numbers, by default False. See the
        notes section for more details.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with an instance numbers column.

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
    count. This is cheaper when you only need to detect a new instance, not
    know it's specifically the first or Nth occurrence.

    Examples
    --------
    Below we add an instance numbers column to a dataset using default
    settings, then use it to analyze a repeated step (e.g., step 5).

    >>> data = amp.Dataset(...)
    >>> ds = add_instance_nums(data)
    >>> step5 = ds[ds['Step'] == 5]
    >>> groups = step5.groupby('InstanceNum')

    Instance numbers also make it easy to pull metrics for just the first
    occurrence of a repeated step within each cycle:

    >>> first_means = step5[step5['InstanceNum'] == 1].groupby('Cycle').mean()

    """
    check_columns = [which, cycle_alias] if cycle_resets else [which]

    _chk._check_columns(data, check_columns)

    ds = data.copy()

    ds[col_name] = _instance_nums(
        data=ds,
        which=which,
        cycle_alias=cycle_alias,
        cycle_resets=cycle_resets,
        fast=fast,
    )

    return ds


def add_relative_time(
    data: Dataset,
    *,
    which: str = 'Step',
    col_name: str = 'StepTime',
    time_alias: str = 'Seconds',
) -> Dataset:
    """
    Add a relative time column to a dataset.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to define groups where relative time resets to zero.
        Defaults to 'Step', resetting relative time at the start of each step.
    col_name : str, optional
        Name of the column to add, by default 'StepTime'.
    time_alias : str, optional
        Name of the column containing global time values, by default 'Seconds'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a relative time column.

    Examples
    --------
    Below we add a relative time column to a dataset using default settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_relative_time(data)

    You may want relative time to reset on changes in State instead of Step,
    e.g., to get relative times for charge/discharge/rest segments that span
    multiple steps (a CCCV charge, or a rest split across steps for varying
    time resolution). Use a different `which` column to control this:

    >>> ds = add_relative_time(data, which='State')

    The time column also doesn't need to be in seconds. If you already have
    another time column, use it via `time_alias`:

    >>> ds['Hours'] = ds['Seconds'] / 3600.0
    >>> ds = add_relative_time(data, time_alias='Hours')

    """
    _chk._check_columns(data, [which, time_alias])

    ds = data.copy()

    instance_nums = _instance_nums(
        data=ds,
        which=which,
        cycle_alias=None,
        cycle_resets=False,
        fast=True,
    )

    groups = ds.groupby([which, instance_nums])[time_alias]
    ds[col_name] = ds[time_alias] - groups.transform('first')

    return ds
