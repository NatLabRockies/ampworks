from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Sequence

from ampworks import _checks as _chk
from ampworks.columns._backend import _instance_nums

if TYPE_CHECKING:
    from ampworks import Dataset


class StepLabel:
    """Label container for a single step."""
    __slots__ = ('label', 'step_num')

    def __init__(self, label: str, step_num: Integral) -> None:
        """
        Mapping steps to human-readable labels can be useful sharing data. Use
        this container to create labels, then apply with `add_step_labels`.
        
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
        Step labels only require a string label and integer step number. They
        can be simple or detailed. Below we define three with varying detail.

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
        Mapping steps to human-readable labels can be useful sharing data. Use
        this container to create labels, then apply with `add_segment_labels`.
        
        We use "segment" here abstractly for describing sections of data defined
        by multiple steps or cycles. These labels can be used to describe cycles
        or for higher-level labels, e.g., reference performance tests (RPTs)
        that can include multiple types of cycles.

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
        `step_nums` or `cycle_nums` must be given, but not both. Use `step_nums`
        to apply a label to all cycles that contain those steps, or `cycle_nums`
        for labeling all steps in the specified cycles.
        
        Examples
        --------
        Below we define segment labels for HPPC and capacity check cycles. A
        higher-level label for RPTs is also defined, which could be added under
        a different column name to indicate the full RPT segment, containing
        both capacity check and HPPC cycles every 50 cycles.

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
    step_alias: str = 'Step',
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
        Name of the column containing step numbers, by default 'Step'.
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
    Using `reset=False` allows step labels to be added to an existing column.
    While this preserves any existing labels, it also allows overwriting labels
    if new steps are given that overlap with existing, already-labeled rows.
    
    Examples
    --------
    Below we create three labels for steps 1, 5, and 6 in the dataset. They are
    then applied using this function. Note that steps not in the list of labels
    are given the default label of 'Unlabeled'.

    >>> data = amp.Dataset(...)
    >>> step1 = StepLabel('Rest', 1)
    >>> step5 = StepLabel('1C CC Charge until 4.2V', 5)
    >>> step6 = StepLabel('CV hold at 4.2V', 6)
    >>> ds = add_step_labels(data, step_labels=[step1, step5, step6])

    """
    _chk._check_columns(data, [step_alias])
    _chk._check_type('step_labels', step_labels, Sequence)
    _chk._check_inner_type('step_labels', step_labels, StepLabel)
    
    ds = data.copy()
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
    step_alias: str = 'Step',
    cycle_alias: str = 'Cycle',
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
        Name of the column containing step numbers, by default 'Step'. Only
        used if some segment labels are defined by step numbers.
    cycle_alias : str, optional
        Name of the column containing cycle numbers, by default 'Cycle'. Only
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
    Using `reset=False` allows segment labels to be added to an existing column.
    While this preserves any existing labels, it also allows overwriting labels
    if new segments are given that overlap with existing, already-labeled rows.
    
    Examples
    --------
    Below we create two new columns for `CycleLabel` and `SegmentLabel`. The
    first column labels all HPPC and capacity check cycles. The second column
    then labels the full RPT segment, which contains both capacity check and
    HPPC cycles, every 50 cycles. Note that since the `CycleLabel` name is used
    twice, we ensure `reset=False` on the second call to avoid losing the first
    set of HPPC labels.

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

    ds = data.copy()

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
    which: str | None = 'Step',
    col_name: str = 'State',
    amps_alias: str = 'Amps',
) -> Dataset:
    """
    Add a state column based on sign of current.
    
    States are 'C' for charge (positive current), 'D' for discharge (negative
    current), and 'R' for rest (zero current). Note that correctly detecting
    rests requires that the current is exactly zero. Consider using a threshold
    to zero-out small currents if this is an issue.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str or None, optional
        The column used to define groups where the state is constant. Defaults
        to 'Step', which assumes each step has a constant state. If None, rows
        are evaluated individually, without groups. See notes for more details.
    col_name : str, optional
        Name of the column to add, by default 'State'
    amps_alias : str, optional
        Name of the column containing current in amps, by default 'Amps'

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a state column.
        
    See Also
    --------
    ~ampworks.Dataset.zero_below : Zero-out small currents to detect rests.
    
    Notes
    -----
    Using `which=None` determines the state for each row individually, without
    considering any groupings. However, this can lead to a single step being
    assigned multiple states if the current changes sign within that step.
    
    The alternative is to use a column that defines groups where the state is
    constant, such as within a step. However, during this grouping, if it is
    determined that the current changes sign within a group, the state is set to
    `'Unknown'` for all rows in that group. This is to avoid assigning incorrect
    states to a group. If you know the state of these unknown rows, you can
    manually set them after adding the state column. See the examples below for
    how to do this.
    
    Examples
    --------
    Below we add a state column to a dataset using default settings. Prior to
    this, the 'Amps' column is zeroed below 1e-8 A to ensure that rests states
    are correctly detected.
    
    >>> data = amp.Dataset(...)
    >>> data = data.zero_below('Amps', threshold=1e-8)
    >>> ds = add_state(data)
    
    Alternative to the thresholding approach, if you know which steps in your
    data are rests, you can zero them out directly by applying a mask to the
    'Amps' column, as demonstrated below.
    
    >>> data = amp.Dataset(...)
    >>> data.loc[data['Step'].isin([1, 3, 11, 13, 21, 23]), 'Amps'] = 0.0
    >>> ds = add_state(data)
    
    You can check to see if any states ended up being assigned the `'Unknown'`
    value, which indicates that the current changed sign within a group by
    filtering.
    
    >>> unknown_states = ds[ds['State'] == 'Unknown']
    >>> print(unknown_states[['Cycle', 'Step']].drop_duplicates())
    
    Once you know which groups have unknown states, you can manually set them to
    the correct value by applying a mask to the state column, as demonstrated.
    
    >>> ds.loc[(ds['Cycle'] == 1) & ds['Step'].isin([5, 7]), 'State'] = 'C'
    >>> ds.loc[(ds['Cycle'] == 2) & ds['Step'].isin([9, 15]), 'State'] = 'D'
    >>> ds.loc[(ds['Cycle'] == 3) & ds['Step'].isin([17, 19]), 'State'] = 'R'
        
    """
    check_columns = [which, amps_alias] if which is not None else [amps_alias]

    _chk._check_columns(data, check_columns)
    
    ds = data.copy()
    
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

        # check if all values in a group are positive, negative, or zero
        all_discharge = groups.transform(lambda x: (x <= 0).all())
        all_charge = groups.transform(lambda x: (x >= 0).all())
        all_zero = groups.transform(lambda x: (x == 0).all())

        ds[col_name] = 'Unknown'
        
        # assign rests last b/c all_charge, all_discharge also true for rests
        ds.loc[all_discharge, col_name] = 'D'
        ds.loc[all_charge, col_name] = 'C'
        ds.loc[all_zero, col_name] = 'R'
    
    ds[col_name] = ds[col_name].astype('category')
    return ds


def add_control_mode(
    data: Dataset,
    *,
    which: str = 'Step',
    col_name: str = 'ControlMode',
    amps_alias: str = 'Amps',
    volts_alias: str = 'Volts',
    watts_alias: str | None = None,
    rtol: float = 5e-3,
    default: str = 'Unknown',
) -> Dataset:
    r"""
    Add a control mode column to a dataset.
    
    Control modes are 'CC' for constant current, 'CV' for constant voltage, 'CP'
    for constant power, or 'Rest' for Rest. The control modes are determined by
    checking segments to see if the current, voltage, and/or power is near
    constant within a given relative tolerance. Rests are detected for segments
    where current is exactly zero. Note that the 'CP' mode is only included when
    `watts_alias` is not `None`.
    
    The near constant check is considered satisfied when the following is true
    for all values in a segment:
    
    .. math::
    
        ({\rm max} - {\rm min}) \le ({\rm rtol} \times {\rm mean})
    
    Parameters
    ----------
    data : Dataset
        The input dataset.
    which : str, optional
        The column used to define groups where control mode is constant. The
        default is 'Step', which assumes each step has a constant control mode.
    col_name : str, optional
        Name of the column to add, by default 'ControlMode'
    amps_alias : str, optional
        Name of the column containing current in amps, by default 'Amps'.
    volts_alias : str, optional
        Name of the column containing voltage in volts, by default 'Volts'.
    watts_alias : str | None, optional
        Name of the column containing power in watts. None (default) disables
        CP mode detection.
    rtol : float, optional
        Relative tolerance for detecting constant current, voltage, or power,
        by default 5e-3.
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
    
    If you want to include 'CP' mode detection, supply the name of the power
    column to `watts_alias`. If needed, you can also add a power column first,
    using :func:`~ampworks.columns.add_power`. In case you are missing a step
    column, use a different column to detect mode changes (e.g., 'State'). A
    state column can also be added if missing from your data using the function
    :func:`~ampworks.columns.add_state`. We demonstrate this case below.
    
    >>> data = amp.Dataset(...)
    >>> ds = add_state(data)
    >>> ds = add_power(ds)
    >>> ds = add_control_mode(ds, which='State', watts_alias='Watts')
    
    """
    check_columns = [which, amps_alias, volts_alias]
    mode_map = {amps_alias: 'CC', volts_alias: 'CV'}
    if watts_alias is not None:
        check_columns.append(watts_alias)
        mode_map[watts_alias] = 'CP'
        
    _chk._check_columns(data, check_columns)
    
    ds = data.copy()
            
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
    mean = groups.transform('mean').abs()

    matches = (maximum - minimum) <= rtol * mean
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
