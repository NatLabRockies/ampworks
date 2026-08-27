from __future__ import annotations

from typing import TYPE_CHECKING

from ampworks import _checks as _chk
from ampworks._core._std_head import SECONDS, CYCLE, STEP
from ampworks.columns._backend import _instance_nums

if TYPE_CHECKING:
    from ampworks import Dataset


def add_instance_nums(
    data: Dataset,
    *,
    which: str = STEP,
    col_name: str = 'InstanceNum',
    cycle_alias: str = CYCLE,
    cycle_resets: bool = True,
    fast: bool = False,
) -> Dataset:
    """
    Add an instance numbers column to a dataset.

    Instance numbers uniquely identify repeated occurrences of a group. For
    example, if a single HPPC cycle contains ten discharge pulses that all
    share the same step number (e.g., 5), instance numbers label them 1 through
    10. Numbering can reset every cycle or stay globally unique, depending on
    `cycle_resets`. Numbering is 1-based, so the first instance of a group is
    always 1, not 0.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to detect and group instances, step-based by default.
    col_name : str, optional
        Name of the column to add, by default 'InstanceNum'.
    cycle_alias : str, optional
        Name of the column containing cycle numbers; defaults to standard name.
        Only used if the `cycle_resets` argument is True.
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
    With `fast=True`, all group resets are ignored and result in a monotonically
    increasing changeover counter, not a per-group occurrence count. This is
    computationally cheaper when you only need to detect a new instance, and do
    not care whether it's specifically the first or Nth occurrence.

    Examples
    --------
    Below we add an instance numbers column to a dataset using default settings,
    then use it to construct groups for downstream analysis on the instances of
    step 5.

    >>> data = amp.Dataset(...)
    >>> ds = add_instance_nums(data)
    >>> step5 = ds[ds['Step'] == 5]
    >>> groups = step5.groupby('InstanceNum')

    Instance numbers can also make it easy to pull metrics/statistics for any
    Nth occurrence of a repeated step within each cycle:

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
    which: str = STEP,
    col_name: str = 'StepTime',
    time_alias: str = SECONDS,
) -> Dataset:
    """
    Add a relative time column to a dataset.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to define groups where relative time resets to zero.
        Defaults to step-based, resetting at the start of each step.
    col_name : str, optional
        Name of the relative time column to add, by default 'StepTime'.
    time_alias : str, optional
        Name of column containing global time values; defaults to the standard
        name for time in seconds.

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
    multiple steps (e.g., a CCCV charge). Use a different `which` column to
    control this:

    >>> ds = add_relative_time(data, which='State')

    The time column also doesn't need to be in seconds. If you have another time
    column, use it via `time_alias`:

    >>> data['Hours'] = data['Seconds'] / 3600.0
    >>> ds = add_relative_time(data, col_name='StepTimeHr', time_alias='Hours')

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
