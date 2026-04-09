import pytest
import pandas as pd
import ampworks as amp


@pytest.fixture(scope='module')
def datasets():
    datasets = {'hppc': amp.datasets.load_datasets('hppc/hppc_discharge')}
    return datasets


def test_step_labels():

    # catches type errors
    with pytest.raises(TypeError):
        _ = amp.StepLabel(1, 1)

    with pytest.raises(TypeError):
        _ = amp.StepLabel('1', '1')

    # correct usage
    _ = amp.StepLabel('1', 1)


def test_cycle_labels():

    # catches type errors
    with pytest.raises(TypeError):
        _ = amp.CycleLabel(1, steps=[1])

    with pytest.raises(TypeError):
        _ = amp.CycleLabel(1, cycles=[1])

    # catch steps vs. cycles conflicts
    with pytest.raises(ValueError):
        _ = amp.CycleLabel('1')

    with pytest.raises(ValueError):
        _ = amp.CycleLabel('1', steps=None, cycles=None)

    with pytest.raises(ValueError):
        _ = amp.CycleLabel('1', steps=[1], cycles=[1])

    # correct usage
    label = amp.CycleLabel('1', steps=[1])
    assert isinstance(label, amp.CycleLabel)
    assert label.label == '1'
    assert label.steps == [1]
    assert label.cycles is None

    label = amp.CycleLabel('1', cycles=[1])
    assert isinstance(label, amp.CycleLabel)
    assert label.label == '1'
    assert label.steps is None
    assert label.cycles == [1]


def test_section_labels():

    # catches type errors
    with pytest.raises(TypeError):
        _ = amp.SectionLabel(1, steps=[1])

    with pytest.raises(TypeError):
        _ = amp.SectionLabel(1, cycles=[1])

    # catch steps vs. cycles conflicts
    with pytest.raises(ValueError):
        _ = amp.SectionLabel('1')

    with pytest.raises(ValueError):
        _ = amp.SectionLabel('1', steps=None, cycles=None)

    with pytest.raises(ValueError):
        _ = amp.SectionLabel('1', steps=[1], cycles=[1])

    # correct usage
    label = amp.SectionLabel('1', steps=[1])
    assert isinstance(label, amp.SectionLabel)
    assert label.label == '1'
    assert label.steps == [1]
    assert label.cycles is None

    label = amp.SectionLabel('1', cycles=[1])
    assert isinstance(label, amp.SectionLabel)
    assert label.label == '1'
    assert label.steps is None
    assert label.cycles == [1]


def test_label_set(datasets):

    step1 = amp.StepLabel('1', 1)
    step2 = amp.StepLabel('2', 2)

    cycle1 = amp.CycleLabel('1', cycles=[1])
    cycle2 = amp.CycleLabel('2', cycles=[2])

    sect1 = amp.SectionLabel('1', cycles=[1])
    sect2 = amp.SectionLabel('2', cycles=[2])

    # catches type errors
    with pytest.raises(TypeError):
        _ = amp.LabelSet(step_labels='error')

    with pytest.raises(TypeError):
        _ = amp.LabelSet(step_labels=[step1, 'error'])

    with pytest.raises(TypeError):
        _ = amp.LabelSet(cycle_labels='error')

    with pytest.raises(TypeError):
        _ = amp.LabelSet(cycle_labels=[cycle1, 'error'])

    with pytest.raises(TypeError):
        _ = amp.LabelSet(section_labels='error')

    with pytest.raises(TypeError):
        _ = amp.LabelSet(section_labels=[sect1, 'error'])

    # correct usage
    labels = amp.LabelSet()
    assert isinstance(labels, amp.LabelSet)
    assert labels.step_labels is None
    assert labels.cycle_labels is None
    assert labels.section_labels is None

    labels = amp.LabelSet(step_labels=[step1, step2])
    assert isinstance(labels, amp.LabelSet)
    assert labels.step_labels == [step1, step2]
    assert labels.cycle_labels is None
    assert labels.section_labels is None

    labels = amp.LabelSet(cycle_labels=[cycle1, cycle2])
    assert isinstance(labels, amp.LabelSet)
    assert labels.step_labels is None
    assert labels.cycle_labels == [cycle1, cycle2]
    assert labels.section_labels is None

    labels = amp.LabelSet(section_labels=[sect1, sect2])
    assert isinstance(labels, amp.LabelSet)
    assert labels.step_labels is None
    assert labels.cycle_labels is None
    assert labels.section_labels == [sect1, sect2]

    labels = amp.LabelSet(
        step_labels=[step1, step2],
        cycle_labels=[cycle1, cycle2],
        section_labels=[sect1, sect2]
    )

    assert isinstance(labels, amp.LabelSet)
    assert labels.step_labels == [step1, step2]
    assert labels.cycle_labels == [cycle1, cycle2]
    assert labels.section_labels == [sect1, sect2]


def test_apply_labels(datasets):
    from ampworks._checks import _check_inner_type

    data1 = datasets['hppc'].copy()

    data2 = data1.copy()
    data2['Cycle'] = 2
    data2['Seconds'] += data2['Seconds'].max() + 1.

    data = pd.concat([data1, data2], ignore_index=True)

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
        section_labels=[
            amp.SectionLabel('RPT1', cycles=[1]),  # cycle 2 defaults to 'None'
        ],
    )

    # label columns were not present before function call
    assert 'StepLabel' not in data.columns
    assert 'CycleLabel' not in data.columns
    assert 'SectionLabel' not in data.columns

    # new dataset with labels, columns added, all values are strings
    labeled = amp.apply_labels(data, labels)

    assert labeled is not data  # new dataset returned

    assert 'StepLabel' in labeled.columns
    assert 'CycleLabel' in labeled.columns
    assert 'SectionLabel' in labeled.columns

    _check_inner_type('StepLabel', labeled['StepLabel'].values, str)
    _check_inner_type('CycleLabel', labeled['CycleLabel'].values, str)
    _check_inner_type('SectionLabel', labeled['SectionLabel'].values, str)

    steps = {
        'Initial Rest',
        'C/3 Discharge',
        'Equilibrium Rest',
        '1C Discharge Pulse',
        '40s Rest',
        '0.75C Charge Pulse',
        'Final Rest',
    }
    assert set(labeled['StepLabel'].unique()) == steps

    cycles = {'HPPC'}
    assert set(labeled['CycleLabel'].unique()) == cycles

    sections = {'RPT1', 'None'}  # cycle 2 defaulted to 'None'
    assert set(labeled['SectionLabel'].unique()) == sections
