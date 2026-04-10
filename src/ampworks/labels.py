"""
A module for applying user-defined labels to steps, cycles, and sections of a
dataset. This is intended to be used for datasets that have a 'Cycle' and 'Step'
column. Note that all classes and functions in this module are also available in
the top-level `ampworks` namespace. This is done for convenience so that calls
to these classes and functions can stay short and simple, e.g. `amp.StepLabel`
instead of `amp.labels.StepLabel`. However, the module is still available
separately to help discoverability and to keep the code organized.

"""
from __future__ import annotations

from numbers import Integral
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset

__all__ = [
    'StepLabel',
    'CycleLabel',
    'SectionLabel',
    'LabelSet',
    'apply_labels',
]


class StepLabel:

    __slots__ = ('label', 'step')

    def __init__(self, label: str, step: Integral) -> None:
        """
        Readable label for a single step.

        Parameters
        ----------
        label : str
            The label to apply for the step, defined by the 'step' argument.
        step : int
            The step number to which the label will be applied.

        Raises
        ------
        TypeError
            If `label` is not a string or if `step` is not an integer.

        Examples
        --------
        The following demonstrates how to create step labels for a dataset. The
        inputs must be a string label and an integer step number. The label can
        be as simple or descriptive as desired.

        >>> from ampworks import StepLabel
        >>> step1 = StepLabel('Initial Rest', 1)
        >>> step2 = StepLabel('1C CC Charge until 4.2V', 2)
        >>> step3 = StepLabel('CV Charge at 4.2V', 3)
        >>> step4 = StepLabel('Final Rest', 4)

        """
        from ampworks._checks import _check_type

        _check_type('label', label, str)
        _check_type('step', step, Integral)

        self.label = label
        self.step = step


class CycleLabel:

    __slots__ = ('label', 'steps', 'cycles')

    def __init__(
        self,
        label: str,
        *,
        steps: Sequence[Integral] | None = None,
        cycles: Sequence[Integral] | None = None,
    ) -> None:
        """
        Readable label for cycles.

        Parameters
        ----------
        label : str
            The label to apply for the cycle, defined by the 'steps' argument.
        steps : Sequence[int] or None, optional
            The step numbers that define the cycle. Defaults to None. Must be
            provided if `cycles` is not provided.
        cycles : Sequence[int] or None, optional
            The cycle numbers that define the cycle. Defaults to None. Must be
            provided if `steps` is not provided.

        Raises
        ------
        TypeError
            If the `label` is not a string or if `steps` or `cycles` is not a
            sequence of integers.
        ValueError
            If both `steps` and `cycles` are `None`, or if both are not `None`.

        Notes
        -----
        Either `steps` or `cycles` must be provided, but not both. If `steps` is
        provided, the label will be applied to all cycles that contain those
        steps. Otherwise, if `cycles` is provided, the label will be applied to
        those cycles.

        Examples
        --------
        The following demonstrates how to create cycle labels for a dataset. The
        inputs must be a string label and either a sequence of step numbers or a
        sequence of cycle numbers. Each label can only be defined using either
        steps or cycles, not both. However, when applying the labels, there are
        no restrictions on mixing cycle labels defined using both methods. Also,
        you can use a `range` object to define your sequence if more convenient,
        so there is no need to explicitly list all step or cycle numbers.

        >>> from ampworks import CycleLabel
        >>> cycle1 = CycleLabel('HPPC', steps=range(1, 9))
        >>> cycle2 = CycleLabel('Aging Cycle', cycles=[1, 2, 3, 4, 5])

        """
        from ampworks._checks import (
            _check_type, _check_inner_type, _check_only_one,
        )

        _check_type('label', label, str)
        _check_only_one(
            conditions=[steps is None, cycles is None],
            message="'steps' and 'cycles' cannot both be 'None'.",
        )
        _check_only_one(
            conditions=[steps is not None, cycles is not None],
            message="'steps' and 'cycles' cannot both be provided.",
        )

        if steps is not None:
            _check_type('steps', steps, Sequence)
            _check_inner_type('steps', steps, Integral)
        elif cycles is not None:
            _check_type('cycles', cycles, Sequence)
            _check_inner_type('cycles', cycles, Integral)

        self.label = label
        self.steps = steps
        self.cycles = cycles


class SectionLabel:

    __slots__ = ('label', 'steps', 'cycles')

    def __init__(
        self,
        label: str,
        *,
        steps: Sequence[Integral] | None = None,
        cycles: Sequence[Integral] | None = None,
    ) -> None:
        """
        Readable label for sections.

        Parameters
        ----------
        label : str
            The label to apply for the section, defined by the 'steps' argument.
        steps : Sequence[int] or None, optional
            The step numbers that define the section. Defaults to None. Must be
            provided if `cycles` is not provided.
        cycles : Sequence[int] or None, optional
            The cycle numbers that define the section. Defaults to None. Must be
            provided if `steps` is not provided.

        Raises
        ------
        TypeError
            If the `label` is not a string or if `steps` or `cycles` is not a
            sequence of integers.
        ValueError
            If both `steps` and `cycles` are `None`, or if both are not `None`.

        Notes
        -----
        Either `steps` or `cycles` must be provided, but not both. If `steps` is
        provided, the label will be applied to all cycles that contain those
        steps. Otherwise, if `cycles` is provided, the label will be applied to
        those cycles.

        Examples
        --------
        The following demonstrates how to create section labels for a dataset. A
        section label is intended to be one level of abstraction higher than a
        cycle label.

        The inputs must be a string label and either a sequence of step numbers
        or a sequence of cycle numbers. Each label can only be defined using
        either steps or cycles, not both. However, when applying the labels,
        there are no restrictions on mixing section labels defined using both
        methods. Also, you can use a `range` object to define your sequence if
        more convenient, so there is no need to list all step or cycle numbers.

        >>> from ampworks import SectionLabel
        >>> section1 = SectionLabel('RPT1', cycles=[1, 2, 3])
        >>> section2 = SectionLabel('RPT2', cycles=range(100, 104))

        """
        from ampworks._checks import (
            _check_type, _check_inner_type, _check_only_one,
        )

        _check_type('label', label, str)
        _check_only_one(
            conditions=[steps is None, cycles is None],
            message="'steps' and 'cycles' cannot both be 'None'.",
        )
        _check_only_one(
            conditions=[steps is not None, cycles is not None],
            message="'steps' and 'cycles' cannot both be provided.",
        )

        if steps is not None:
            _check_type('steps', steps, Sequence)
            _check_inner_type('steps', steps, Integral)
        elif cycles is not None:
            _check_type('cycles', cycles, Sequence)
            _check_inner_type('cycles', cycles, Integral)

        self.label = label
        self.steps = steps
        self.cycles = cycles


class LabelSet:

    __slots__ = ('step_labels', 'cycle_labels', 'section_labels')

    def __init__(
        self,
        *,
        step_labels: Sequence[StepLabel] | None = None,
        cycle_labels: Sequence[CycleLabel] | None = None,
        section_labels: Sequence[SectionLabel] | None = None,
    ) -> None:
        """
        A set of step, cycle, and section labels.

        Parameters
        ----------
        step_labels : Sequence[StepLabel] or None, optional
            The step labels for a given dataset. Defaults to None.
        cycle_labels : Sequence[CycleLabel] or None, optional
            The cycle labels for a given dataset. Defaults to None.
        section_labels : Sequence[SectionLabel] or None, optional
            The section labels for a given dataset. Defaults to None.

        Raises
        ------
        TypeError
            If any of the label inputs are not sequences of the appropriate
            label type, e.g. if `step_labels` is not a sequence of `StepLabel`.

        """
        from ampworks._checks import _check_type, _check_inner_type

        if step_labels is not None:
            _check_type('step_labels', step_labels, Sequence)
            _check_inner_type('step_labels', step_labels, StepLabel)
        if cycle_labels is not None:
            _check_type('cycle_labels', cycle_labels, Sequence)
            _check_inner_type('cycle_labels', cycle_labels, CycleLabel)
        if section_labels is not None:
            _check_type('section_labels', section_labels, Sequence)
            _check_inner_type('section_labels', section_labels, SectionLabel)

        self.step_labels = step_labels
        self.cycle_labels = cycle_labels
        self.section_labels = section_labels


def apply_labels(data: Dataset, labels: LabelSet) -> Dataset:
    """
    Apply labels to a dataset.

    Parameters
    ----------
    data : Dataset
        The dataset to which labels will be applied.
    labels : LabelSet
        The set of step, cycle, and section labels to apply.

    Returns
    -------
    labeled : Dataset
        A new dataset with the applied labels.

    Raises
    ------
    TypeError
        If `data` is not a Dataset or if `labels` is not a LabelSet.
    ValueError
        If `data` does not contain the required 'Cycle' and 'Step' columns.

    Notes
    -----
    Any cycles or steps that are not labeled will have a label of `'None'` in
    the resulting `CycleLabel` and `StepLabel` columns. Also, if a step is given
    in more than one cycle, the last cycle label will be applied. Similarly, if
    a step is given more than one step label, only the last step label will be
    applied to that step.

    Examples
    --------
    The following demonstrates how to apply labels to an HPPC dataset from the
    `ampworks.datasets` module. First, we create the labels and them apply them
    to the dataset. The final line of code plots the resulting dataset with some
    hover hints so the applied labels can be viewed and checked for accuracy.

    .. code-block:: python

        import ampworks as amp

        # Load in an example dataset
        data = amp.datasets.load_datasets('hppc/hppc_discharge')

        # Create the labels
        labels = amp.LabelSet(
            step_labels=[
                amp.StepLabel('Initial Rest', 1),
                amp.StepLabel('C/3 Discharge', 2),
                amp.StepLabel('Equilibrium Rest', 3),
                amp.StepLabel('1C Discharge Pulse', 4),
                amp.StepLabel('40s Rest', 5),
                amp.StepLabel('0.75C Charge Pulse', 6),
                amp.StepLabel('40s Rest', 7),
                amp.StepLabel('Final Rest', 8),
            ],
            cycle_labels=[
                amp.CycleLabel('HPPC', steps=range(1, 9)),
            ],
        )

        # Apply the labels
        labeled = amp.apply_labels(all_data, labels)

        # Add an hours column and plot with the labels as hover tips
        labeled['Hours'] = labeled['Seconds'] / 3600

        labeled.interactive_xy_plot(
            x='Hours', y='Volts', tips=['StepLabel', 'CycleLabel'],
        )

    """
    from ampworks import Dataset
    from ampworks._checks import _check_type, _check_columns

    _check_type('data', data, Dataset)
    _check_type('labels', labels, LabelSet)
    _check_columns(data, ['Cycle', 'Step'])

    # Create a copy of the dataset to avoid modifying the original
    labeled = data.copy()

    def _which_to_label(
        which_label: CycleLabel | StepLabel,
    ) -> tuple[str, Sequence[Integral], str]:

        if which_label.steps is not None:
            name, cond = 'Step', which_label.steps

        elif which_label.cycles is not None:
            name, cond = 'Cycle', which_label.cycles

        return name, cond, which_label.label

    # Apply step labels
    if labels.step_labels is not None:

        labeled['StepLabel'] = 'None'  # Initialize the column
        for step_label in labels.step_labels:
            label, step = step_label.label, step_label.step
            labeled.loc[labeled['Step'] == step, 'StepLabel'] = label

    # Apply cycle labels
    if labels.cycle_labels is not None:

        labeled['CycleLabel'] = 'None'  # Initialize the column
        for cycle_label in labels.cycle_labels:
            name, cond, label = _which_to_label(cycle_label)
            labeled.loc[labeled[name].isin(cond), 'CycleLabel'] = label

    # Apply section labels
    if labels.section_labels is not None:

        labeled['SectionLabel'] = 'None'  # Initialize the column
        for section_label in labels.section_labels:
            name, cond, label = _which_to_label(section_label)
            labeled.loc[labeled[name].isin(cond), 'SectionLabel'] = label

    return labeled
