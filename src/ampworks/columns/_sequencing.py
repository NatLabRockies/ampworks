from __future__ import annotations

from ampworks import _checks as _chk
from ampworks._core._dataset import Dataset
from ampworks.columns._backend import _instance_nums
from ampworks._core._std_names import SECONDS, CYCLE, STEP


def add_instance_nums(
    data: Dataset,
    *,
    by: str = STEP,
    output_col: str = 'InstanceNum',
    cycle_col: str = CYCLE,
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
    by : str, optional
        The column used to detect and group instances, step-based by default.
    output_col : str, optional
        Name of instance numbers column to add, by default 'InstanceNum'.
    cycle_col : str, optional
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
        If `cycle_resets` is True and `cycle_col` is None.

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
    check_columns = [by, cycle_col] if cycle_resets else [by]

    if cycle_resets and (cycle_col is None):
        raise ValueError("'cycle_col' is required when cycle_resets=True.")

    _chk._check_columns(data, check_columns)

    ds = Dataset(data)

    ds[output_col] = _instance_nums(
        data=ds,
        by=by,
        cycle_col=cycle_col,
        cycle_resets=cycle_resets,
        fast=fast,
    )

    return ds


def add_relative_time(
    data: Dataset,
    *,
    by: str = STEP,
    output_col: str = 'StepTime',
    time_col: str = SECONDS,
) -> Dataset:
    """
    Add a relative time column to a dataset.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    by : str, optional
        The column used to define groups where relative time resets to zero.
        Defaults to step-based, resetting at the start of each step.
    output_col : str, optional
        Name of the relative time column to add, by default 'StepTime'.
    time_col : str, optional
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
    multiple steps (e.g., a CCCV charge). Use a different `by` column to
    control this:

    >>> ds = add_relative_time(data, by='State')

    The time column also doesn't need to be in seconds. If you have another time
    column, use it via `time_col`:

    >>> data['Hours'] = data['Seconds'] / 3600.0
    >>> ds = add_relative_time(data, output_col='StepTimeHr', time_col='Hours')

    """
    _chk._check_columns(data, [by, time_col])

    ds = Dataset(data)

    instance_nums = _instance_nums(
        data=ds,
        by=by,
        cycle_col=None,
        cycle_resets=False,
        fast=True,
    )

    groups = ds.groupby([by, instance_nums])[time_col]
    ds[output_col] = ds[time_col] - groups.transform('first')

    return ds
