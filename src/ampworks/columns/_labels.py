from __future__ import annotations

from numbers import Integral, Real
from typing import Dict, Sequence

import pandas as pd

from ampworks import _checks as _chk
from ampworks._core._dataset import Dataset
from ampworks.columns._backend import _instance_nums
from ampworks._core._std_head import AMPS, VOLTS, CYCLE, STEP, STATE


class StepLabel:
    """Label container for a single step."""

    __slots__ = ('label', 'step_num')

    def __init__(self, label: str, step_num: Integral) -> None:
        """
        Container mapping a step number to a human-readable label, for use with
        `add_step_labels`.

        Parameters
        ----------
        label : str
            The label for a given step.
        step_num : int
            Step number associated with the label.

        See Also
        --------
        ~ampworks.columns.add_step_labels : Add step labels to a dataset.

        Examples
        --------
        Labels can be as simple or detailed as needed:

        >>> step1 = StepLabel('Rest', 1)
        >>> step2 = StepLabel('1C CC Charge until 4.2V', 2)
        >>> step3 = StepLabel('CV hold at 4.2V', 3)

        """
        _chk._check_type('label', label, str)
        _chk._check_type('step_num', step_num, Integral)

        self.label = label
        self.step_num = step_num

    def __repr__(self) -> str:
        return f"StepLabel(label={self.label}, step_num={self.step_num})"


class SegmentLabel:
    """Label container for segments of steps or cycles."""

    __slots__ = ('label', 'step_nums', 'cycle_nums')

    def __init__(
        self,
        label: str,
        *,
        step_nums: Sequence[Integral] | None = None,
        cycle_nums: Sequence[Integral] | None = None,
    ) -> None:
        """
        Container mapping step or cycle numbers to a human-readable label, for
        use with `add_segment_labels`. "Segment" refers broadly to any section
        defined by steps or cycles, e.g., an RPT spanning several cycle types.

        Parameters
        ----------
        label : str
            The label for a given segment.
        step_nums : Sequence[int] or None, optional
            Step number(s) associated with the label, by default None. You must
            provide either `step_nums` or `cycle_nums`, not both.
        cycle_nums : Sequence[int] or None, optional
            Cycle number(s) associated with the label, by default None. You must
            provide either `step_nums` or `cycle_nums`, not both.

        See Also
        --------
        ~ampworks.columns.add_segment_labels : Add segment labels to a dataset.

        Notes
        -----
        Exactly one of `step_nums`/`cycle_nums` is required. `step_nums` labels
        every cycle containing those steps; `cycle_nums` labels every step
        within those cycles.

        Examples
        --------
        Labels for HPPC and capacity-check cycles, plus a higher-level RPT
        label (e.g., under a separate column) spanning both every 50 cycles:

        >>> hppcs = SegmentLabel('HPPC', step_nums=range(4, 12))
        >>> cap_checks = SegmentLabel('Capacity Check', step_nums=[1, 2])
        >>> rpts = SegmentLabel('RPT', cycle_nums=[1, 2, 52, 53, 103, 104])

        """
        _chk._check_type('label', label, str)
        _chk._check_only_one(
            conditions=[step_nums is not None, cycle_nums is not None],
            message="Provide only one of 'step_nums' or 'cycle_nums'.",
        )

        if step_nums is not None:
            _chk._check_type('step_nums', step_nums, Sequence)
            _chk._check_inner_type('step_nums', step_nums, Integral)
        else:
            _chk._check_type('cycle_nums', cycle_nums, Sequence)
            _chk._check_inner_type('cycle_nums', cycle_nums, Integral)

        self.label = label
        self.step_nums = step_nums
        self.cycle_nums = cycle_nums

    def __repr__(self) -> str:
        name = 'step_nums' if self.step_nums is not None else 'cycle_nums'
        val = self.step_nums if self.step_nums is not None else self.cycle_nums
        return f"SegmentLabel(label={self.label}, {name}={val})"


def add_step_labels(
    data: Dataset,
    *,
    step_labels: Sequence[StepLabel],
    col_name: str = 'StepLabel',
    step_alias: str = STEP,
    reset: bool = False,
    default: str = 'Unlabeled',
) -> Dataset:
    """
    Add step labels to a dataset.

    Use `StepLabel` to construct readable labels for some (or all) steps in your
    data and then apply them with this function.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    step_labels : Sequence[StepLabel]
        A sequence of `StepLabel` defining the map of labels to step numbers.
    col_name : str, optional
        Name of the column to add, by default 'StepLabel'.
    step_alias : str, optional
        Name of column containing step numbers; defaults to standard name.
    reset : bool, optional
        Whether to reset the column before adding labels, by default False.
    default : str, optional
        Label for steps not explicitly labeled, by default 'Unlabeled'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a step labels column.

    See Also
    --------
    ~ampworks.columns.StepLabel : Container to define step labels.

    Notes
    -----
    With `reset=False`, new labels are layered onto any existing column,
    overwriting only the steps you relabel.

    Examples
    --------
    Steps not included in `step_labels` get the `default` label:

    >>> data = amp.Dataset(...)
    >>> step1 = StepLabel('Rest', 1)
    >>> step5 = StepLabel('1C CC Charge until 4.2V', 5)
    >>> step6 = StepLabel('CV hold at 4.2V', 6)
    >>> ds = add_step_labels(data, step_labels=[step1, step5, step6])

    """
    _chk._check_columns(data, [step_alias])
    _chk._check_type('step_labels', step_labels, Sequence)
    _chk._check_inner_type('step_labels', step_labels, StepLabel)

    ds = Dataset(data)
    ds[step_alias] = ds[step_alias].astype(int)

    if reset or (col_name not in ds.columns):
        ds[col_name] = default
    else:
        # avoid restrictive existing categories from a prior call
        ds[col_name] = ds[col_name].astype(object)

    for s in step_labels:
        ds.loc[ds[step_alias] == s.step_num, col_name] = s.label

    ds[col_name] = ds[col_name].astype('category')
    return ds


def add_segment_labels(
    data: Dataset,
    *,
    segment_labels: Sequence[SegmentLabel],
    col_name: str = 'SegmentLabel',
    step_alias: str = STEP,
    cycle_alias: str = CYCLE,
    reset: bool = False,
    default: str = 'Unlabeled',
) -> Dataset:
    """
    Add segment labels to a dataset.

    Use `SegmentLabel` to construct readable labels for some (or all) segments
    in your data and then apply them with this function.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    segment_labels : Sequence[SegmentLabel]
        A sequence of `SegmentLabel` defining the map of labels to segments.
    col_name : str, optional
        Name of the column to add, by default 'SegmentLabel'.
    step_alias : str, optional
        Name of column containing step numbers; defaults to standard name. Only
        used if some segment labels are defined by step numbers.
    cycle_alias : str, optional
        Name of column containing cycle numbers; defaults to standard name. Only
        used if some segment labels are defined by cycle numbers.
    reset : bool, optional
        Whether to reset the column before adding labels, by default False.
    default : str, optional
        Label for segments not explicitly labeled, by default 'Unlabeled'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a segments labels column.

    See Also
    --------
    ~ampworks.columns.SegmentLabel : Container to define segment labels.

    Notes
    -----
    With `reset=False`, new labels are layered onto any existing column,
    overwriting only the segments you relabel.

    Examples
    --------
    Two new columns are created below: `CycleLabel` labels HPPC and capacity
    check cycles, while `SegmentLabel` labels the full RPT (both cycle types,
    every 50 cycles). `reset=False` on the second call preserves the HPPC
    labels already written to `CycleLabel`.

    >>> data = amp.Dataset(...)
    >>> hppcs = SegmentLabel('HPPC', step_nums=range(4, 12))
    >>> cap_checks = SegmentLabel('Capacity Check', step_nums=[1, 2])
    >>> rpts = SegmentLabel('RPT', cycle_nums=[1, 2, 52, 53, 103, 104])
    >>> ds = add_segment_labels(
    ...    data, segment_labels=[hppcs], col_name='CycleLabel',
    ... )
    >>> ds = add_segment_labels(
    ...    ds, segment_labels=[cap_checks], col_name='CycleLabel', reset=False,
    ... )
    >>> ds = add_segment_labels(
    ...    ds, segment_labels=[rpts], col_name='SegmentLabel',
    ... )

    """
    _chk._check_type('segment_labels', segment_labels, Sequence)
    _chk._check_inner_type('segment_labels', segment_labels, SegmentLabel)

    ds = Dataset(data)

    needs_step_alias = any(s.step_nums is not None for s in segment_labels)
    needs_cycle_alias = any(s.cycle_nums is not None for s in segment_labels)

    if needs_step_alias:
        _chk._check_columns(data, [step_alias])
        ds[step_alias] = ds[step_alias].astype(int)
    if needs_cycle_alias:
        _chk._check_columns(data, [cycle_alias])
        ds[cycle_alias] = ds[cycle_alias].astype(int)

    if reset or (col_name not in ds.columns):
        ds[col_name] = default
    else:
        # avoid restrictive existing categories from a prior call
        ds[col_name] = ds[col_name].astype(object)

    for s in segment_labels:
        which_alias = step_alias if s.step_nums is not None else cycle_alias
        which_nums = s.step_nums if s.step_nums is not None else s.cycle_nums

        ds.loc[ds[which_alias].isin(which_nums), col_name] = s.label

    ds[col_name] = ds[col_name].astype('category')
    return ds


def add_state(
    data: Dataset,
    *,
    which: str | None = STEP,
    col_name: str = STATE,
    amps_alias: str = AMPS,
    default: str = 'Unknown',
) -> Dataset:
    """
    Add a state column based on current.

    States are 'C' (charge, current > 0), 'D' (discharge, current < 0), or 'R'
    (rest, current == 0). Detecting rests requires exactly zero current. Use a
    threshold to zero out small noise if needed.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str or None, optional
        The column used to define groups where the state is constant. Defaults
        to step-based, which assumes each step has a constant state. If None,
        rows are treated individually, without groups. See notes for more info.
    col_name : str, optional
        Name of state column to add; defaults to standard name.
    amps_alias : str, optional
        Name of column containing current in amps; defaults to standard name.
    default : str, optional
        Default state for rows that do not match any conditions, by default
        'Unknown'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a state column.

    See Also
    --------
    ~ampworks.Dataset.zero_below : Zero-out small currents to detect rests.

    Notes
    -----
    With `which=None`, state is set per row, which can split a single step
    across multiple states if current changes sign mid-step. Grouping by
    `which` avoids that, but assigns `default` to any groups whose current
    changes signs. Relabel these manually if needed (see Examples). Be aware
    that detection of state within groups allows for tapering current to zero,
    so constant-voltage steps that have a constant charge or discharge state,
    but have currents that taper to zero, are still correctly detected.

    Examples
    --------
    Zero out noise before adding a state column so rests are detected correctly:

    >>> data = amp.Dataset(...)
    >>> data = data.zero_below('Amps', threshold=1e-8)
    >>> ds = add_state(data)

    Any group whose current changes sign is labeled by `default`, in this case,
    `'Unknown'`. Find and relabel these manually if you know the correct states:

    >>> unknown = ds[ds['State'] == 'Unknown'][['Cycle', 'Step']]
    >>> print(unknown.drop_duplicates())
    >>> ds.loc[(ds['Cycle'] == 1) & ds['Step'].isin([5, 7]), 'State'] = 'C'

    """
    check_columns = [which, amps_alias] if which is not None else [amps_alias]

    _chk._check_columns(data, check_columns)

    ds = Dataset(data)

    if which is None:
        ds[col_name] = 'R'
        ds.loc[ds[amps_alias] > 0, col_name] = 'C'
        ds.loc[ds[amps_alias] < 0, col_name] = 'D'

    else:
        instance_nums = _instance_nums(
            data=ds,
            which=which,
            cycle_alias=None,
            cycle_resets=False,
            fast=True,
        )

        groups = ds.groupby([which, instance_nums])[amps_alias]

        # a group is 'C'/'D' when its min/max never cross zero; both bounds
        # sitting exactly at zero implies every value in the group is zero
        group_min = groups.transform('min')
        group_max = groups.transform('max')

        all_charge = group_min >= 0
        all_discharge = group_max <= 0
        all_zero = all_charge & all_discharge

        ds[col_name] = default  # default first, then overwrite with matches

        # assign rests last b/c all_charge, all_discharge also true for rests
        ds.loc[all_discharge, col_name] = 'D'
        ds.loc[all_charge, col_name] = 'C'
        ds.loc[all_zero, col_name] = 'R'

    ds[col_name] = ds[col_name].astype('category')
    return ds


def add_control_mode(
    data: Dataset,
    *,
    which: str = STEP,
    col_name: str = 'ControlMode',
    amps_alias: str = AMPS,
    volts_alias: str = VOLTS,
    watts_alias: str | None = None,
    rtol: float | Dict[str, float] = 5e-3,
    default: str = 'Unknown',
) -> Dataset:
    r"""
    Add a control mode column to a dataset.

    Control modes are 'CC', 'CV', 'CP' (only checked if `watts_alias` is given),
    or 'Rest' (current exactly zero). A mode is assigned when its signal is
    constant within `rtol` over a group. If there are no matches or multiple
    matches, the `default` is used instead.

    A signal is "constant" within a group when:

    .. math::

        ({\rm max} - {\rm min}) \le ({\rm rtol} \times {\rm mean})

    where max, min, and mean are in reference to current, voltage, and power to
    detect 'CC', 'CV', and 'CP' (optionally), respectively.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to define groups where control mode is constant. The
        default is step-based, which assumes each step has a constant mode.
    col_name : str, optional
        Name of the column to add, by default 'ControlMode'
    amps_alias : str, optional
        Name of column containing current in amps; defaults to standard name.
    volts_alias : str, optional
        Name of column containing voltage in volts; defaults to standard name.
    watts_alias : str | None, optional
        Name of column containing power in watts; None (default) disables CP
        mode detection.
    rtol : float or Dict[str, float], optional
        Relative tolerance for detecting constant current, voltage, or power,
        by default 5e-3. To use different tolerances for each mode, pass a
        dictionary with keys 'CC', 'CV', and/or 'CP' and their corresponding
        tolerances. In this case needed, but missing keys raise a KeyError.
    default : str, optional
        Default control mode for rows that do not match any mode, or that match
        match more than one, by default 'Unknown'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a control mode column.

    See Also
    --------
    ~ampworks.columns.add_state : Add a state column using the sign of current.
    ~ampworks.columns.add_power : Add a power column using current and voltage.

    Examples
    --------
    Below we add a control mode column to a dataset using default settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_control_mode(data)

    For 'CP' detection, add a `Watts` column and pass it as `watts_alias`. If
    steps are unavailable, group by another constant-per-segment column instead,
    such as state:

    >>> data = amp.Dataset(...)
    >>> ds = add_state(data)
    >>> ds = add_power(ds)
    >>> ds = add_control_mode(ds, which='State', watts_alias='Watts')

    """
    check_columns = [which, amps_alias, volts_alias]
    mode_map = {amps_alias: 'CC', volts_alias: 'CV'}

    if isinstance(rtol, Real):
        rtol = {'CC': rtol, 'CV': rtol, 'CP': rtol}

    if watts_alias is not None:
        check_columns.append(watts_alias)
        mode_map[watts_alias] = 'CP'

    _chk._check_columns(data, check_columns)

    ds = Dataset(data)

    instance_nums = _instance_nums(
        data=ds,
        which=which,
        cycle_alias=None,
        cycle_resets=False,
        fast=True,
    )

    groups = ds.groupby([which, instance_nums])[list(mode_map.keys())]

    minimum = groups.transform('min')
    maximum = groups.transform('max')
    mean = groups.transform('mean')

    # convert rtol to a series for broadcasting to all columns
    rtol_series = pd.Series(mode_map).map(rtol)
    if rtol_series.isnull().any():
        missing = set(mode_map.values()) - set(rtol.keys())
        raise KeyError(f"Missing keys in rtol dictionary: {missing}")

    matches = (maximum - minimum) <= (rtol_series * mean).abs()
    matches = matches.rename(columns=mode_map)

    # take column name of match, ignoring rows with multiple matches
    count = matches.sum(axis=1)  # number of matches per row
    ds[col_name] = matches.idxmax(axis=1).where(count == 1, other=default)

    # override rests - where multiple conditions exist, but have meaning
    rests = (
        ds[amps_alias]
        .eq(0.0)
        .groupby([ds[which], instance_nums])
        .transform('all')
    )

    ds.loc[rests, col_name] = 'Rest'

    ds[col_name] = ds[col_name].astype('category')
    return ds
