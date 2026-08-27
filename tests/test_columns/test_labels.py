import pytest
import numpy as np
import pandas as pd

import ampworks as amp
import ampworks.columns as col


# fmt: off

@pytest.fixture
def label_data():
    """Three steps (1, 2, 3), two rows each, across two cycles."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 3, 3],
        'Cycle': [1, 1, 1, 1, 2, 2],
        'ExpectedStep': ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled'],
        'ExpectedSeg1': ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled'],
        'ExpectedSeg2': ['A', 'A', 'A', 'A', 'Unlabeled', 'Unlabeled'],
        'ExpectedSeg3': ['A', 'A', 'Unlabeled', 'Unlabeled', 'B', 'B'],
    })


@pytest.fixture
def state_data():
    """Steps demonstrating charge, discharge, rest, taper, and mixed sign."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'Amps': [
             1.,  1.,  1.,  # all positive -> charge
            -1., -1., -1.,  # all negative -> discharge
             0.,  0.,  0.,  # exactly zero -> rest
             1.,  0.5, 0.,  # tapers to zero, stays >= 0 -> charge
            -1.,  0.,  1.,  # truly mixed sign -> unknown
        ],
        'ExpectedStep': [   # which='Step' (single state across each step)
            'C', 'C', 'C',
            'D', 'D', 'D',
            'R', 'R', 'R',
            'C', 'C', 'C',
            'Unknown', 'Unknown', 'Unknown',
        ],
        'ExpectedNone': [    # which=None (row-by-row state)
            'C', 'C', 'C',
            'D', 'D', 'D',
            'R', 'R', 'R',
            'C', 'C', 'R',
            'D', 'R', 'C',
        ],
    })


@pytest.fixture
def control_data():
    """Steps demonstrating CC, CV, rest, an ambiguous case, and CP."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
        'Amps': [
             1.0,  1.0,  1.0,  # constant current (charge) -> CC
            -1.0, -1.0, -1.0,  # constant current (discharge) -> CC
             1.0,  0.5,  0.1,  # varying current, constant voltage -> CV
             0.0,  0.0,  0.0,  # exact rest -> Rest (overrides any match)
             1.0,  2.0,  3.0,  # varying current and voltage -> Unknown
             1.0,  2.0,  4.0,  # varying, but constant power -> CP or Unknown
             1.0,  1.0,  1.0,  # ambiguous, meets CC and CV -> Unknown
        ],
        'Volts': [
            3.0, 3.5, 4.0,
            4.0, 3.5, 3.0,
            4.2, 4.2, 4.2,
            4.1, 4.1, 4.0,
            3.0, 3.5, 4.0,
            4.0, 2.0, 1.0,
            4.2, 4.2, 4.2,
        ],
        'Expected': [
            'CC', 'CC', 'CC',
            'CC', 'CC', 'CC',
            'CV', 'CV', 'CV',
            'Rest', 'Rest', 'Rest',
            'Unknown', 'Unknown', 'Unknown',
            'CP', 'CP', 'CP',
            'Unknown', 'Unknown', 'Unknown',
        ],
    })

# fmt: on


class TestStepLabel:
    def test_type_errors(self):
        with pytest.raises(TypeError):  # label should be str
            col.StepLabel(label=1, step_num=1)

        with pytest.raises(TypeError):  # step_num should be int
            col.StepLabel(label='Rest', step_num='1')

    def test_init(self):
        label = col.StepLabel('Rest', 1)
        assert label.label == 'Rest'
        assert label.step_num == 1

    def test_repr(self):
        label = col.StepLabel('Rest', 1)
        assert repr(label) == 'StepLabel(label=Rest, step_num=1)'


class TestSegmentLabel:
    def test_type_errors(self):
        with pytest.raises(TypeError):  # label should be str
            col.SegmentLabel(1, step_nums=[1])

        # step_nums should be Sequence[int]
        with pytest.raises(TypeError):
            col.SegmentLabel('HPPC', step_nums=1)

        with pytest.raises(TypeError):
            col.SegmentLabel('HPPC', step_nums=['1', '2'])

        # cycle_nums should be Sequence[int]
        with pytest.raises(TypeError):
            col.SegmentLabel('RPT', cycle_nums=1)

        with pytest.raises(TypeError):
            col.SegmentLabel('RPT', cycle_nums=['1', '2'])

    def test_requires_one_of_step_or_cycle_nums(self):
        with pytest.raises(ValueError):  # neither given
            col.SegmentLabel('HPPC')

        with pytest.raises(ValueError):  # both given
            col.SegmentLabel('HPPC', step_nums=[1], cycle_nums=[1])

    def test_init_with_step_nums(self):
        label = col.SegmentLabel('HPPC', step_nums=[4, 5, 6])
        assert label.label == 'HPPC'
        assert label.step_nums == [4, 5, 6]
        assert label.cycle_nums is None

    def test_init_with_cycle_nums(self):
        label = col.SegmentLabel('RPT', cycle_nums=[1, 2])
        assert label.label == 'RPT'
        assert label.cycle_nums == [1, 2]
        assert label.step_nums is None

    def test_repr(self):
        step_label = col.SegmentLabel('HPPC', step_nums=[1, 2])
        cycle_label = col.SegmentLabel('RPT', cycle_nums=[1, 2])

        assert repr(step_label) == 'SegmentLabel(label=HPPC, step_nums=[1, 2])'
        assert repr(cycle_label) == 'SegmentLabel(label=RPT, cycle_nums=[1, 2])'


class TestAddStepLabels:
    step_labels = [
        col.StepLabel('A', 1),
        col.StepLabel('B', 2),
    ]

    def test_keyword_only(self, label_data):
        with pytest.raises(TypeError, match='positional'):
            col.add_step_labels(label_data, self.step_labels)

    def test_missing_step_column_raises(self, label_data):
        ds = label_data.rename(columns={'Step': 'StepAlias'})
        with pytest.raises(ValueError):
            col.add_step_labels(ds, step_labels=self.step_labels)

    def test_requires_step_label_instances(self, label_data):
        with pytest.raises(TypeError):
            col.add_step_labels(label_data, step_labels=[('A', 1)])

    def test_step_nums_labeling(self, label_data):
        result = col.add_step_labels(label_data, step_labels=self.step_labels)

        expected = label_data['ExpectedStep']
        assert result['StepLabel'].astype(str).equals(expected)

    def test_col_name(self, label_data):
        result = col.add_step_labels(
            label_data,
            step_labels=self.step_labels,
            col_name='Custom',
        )
        assert 'Custom' in result.columns
        assert 'StepLabel' not in result.columns

    def test_step_alias(self, label_data):
        result = col.add_step_labels(
            label_data.rename(columns={'Step': 'StepAlias'}),
            step_labels=self.step_labels,
            step_alias='StepAlias',
        )

        expected = label_data['ExpectedStep']
        assert result['StepLabel'].astype(str).equals(expected)

    def test_custom_default_labels(self, label_data):
        result = col.add_step_labels(
            label_data,
            step_labels=self.step_labels,
            default='None',
        )

        expected = label_data['ExpectedStep'].replace({'Unlabeled': 'None'})

        assert 'None' in expected.to_list()
        assert result['StepLabel'].astype(str).equals(expected)

    def test_reset_false_accumulates_labels(self, label_data):
        first = col.add_step_labels(
            label_data,
            step_labels=self.step_labels[:1],
        )
        second = col.add_step_labels(
            first,
            step_labels=self.step_labels[1:],
            reset=False,
        )

        expected = label_data['ExpectedStep']
        assert second['StepLabel'].astype(str).equals(expected)

    def test_reset_true_clears_labels(self, label_data):
        first = col.add_step_labels(
            label_data,
            step_labels=self.step_labels[:1],
        )
        assert 'A' in first['StepLabel'].astype(str).tolist()

        second = col.add_step_labels(
            first,
            step_labels=self.step_labels[1:],
            reset=True,
        )
        assert 'A' not in second['StepLabel'].astype(str).tolist()

        expected = label_data['ExpectedStep'].replace({'A': 'Unlabeled'})
        assert second['StepLabel'].astype(str).equals(expected)

    def test_returns_copy(self, label_data):
        result = col.add_step_labels(label_data, step_labels=self.step_labels)
        assert result is not label_data
        assert 'StepLabel' not in label_data.columns

    def test_output_is_categorical(self, label_data):
        result = col.add_step_labels(label_data, step_labels=self.step_labels)
        assert isinstance(result['StepLabel'].dtype, pd.CategoricalDtype)


class TestAddSegmentLabels:
    step_labels1 = [
        col.SegmentLabel('A', step_nums=[1]),
        col.SegmentLabel('B', step_nums=[2]),
    ]

    step_labels2 = [col.SegmentLabel('A', step_nums=[1, 2])]
    cycle_labels2 = [col.SegmentLabel('A', cycle_nums=[1])]

    step_labels3 = [
        col.SegmentLabel('A', step_nums=[1]),
        col.SegmentLabel('B', step_nums=[3]),
    ]

    mixed_labels3 = [
        col.SegmentLabel('A', step_nums=[1]),
        col.SegmentLabel('B', cycle_nums=[2]),
    ]

    def test_keyword_only(self, label_data):
        with pytest.raises(TypeError, match='positional'):
            col.add_segment_labels(label_data, self.step_labels1)

    def test_missing_step_column_raises(self, label_data):
        ds = label_data.drop(columns=['Step'])

        # only raises when segment_labels contains step_nums
        with pytest.raises(ValueError):
            col.add_segment_labels(ds, segment_labels=self.step_labels1)

        # no errors when segment_labels contains only cycle_nums
        result = col.add_segment_labels(ds, segment_labels=self.cycle_labels2)
        assert 'SegmentLabel' in result.columns

    def test_missing_cycle_column_raises(self, label_data):
        ds = label_data.drop(columns=['Cycle'])

        # only raises when segment_labels contains cycle_nums
        with pytest.raises(ValueError):
            col.add_segment_labels(ds, segment_labels=self.cycle_labels2)

        # no errors when segment_labels contains only step_nums
        result = col.add_segment_labels(ds, segment_labels=self.step_labels2)
        assert 'SegmentLabel' in result.columns

    def test_requires_segment_label_instances(self, label_data):
        with pytest.raises(TypeError):
            col.add_segment_labels(label_data, segment_labels=[('X', [1])])

    def test_step_nums_labeling(self, label_data):
        result1 = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1,
        )

        expected1 = label_data['ExpectedSeg1']
        assert result1['SegmentLabel'].astype(str).equals(expected1)

        result2 = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels2,
        )

        expected2 = label_data['ExpectedSeg2']
        assert result2['SegmentLabel'].astype(str).equals(expected2)

    def test_cycle_nums_labeling(self, label_data):
        result = col.add_segment_labels(
            label_data,
            segment_labels=self.cycle_labels2,
        )

        expected = label_data['ExpectedSeg2']
        assert result['SegmentLabel'].astype(str).equals(expected)

    def test_mixed_step_and_cycle_labels(self, label_data):
        result = col.add_segment_labels(
            label_data,
            segment_labels=self.mixed_labels3,
        )

        expected = label_data['ExpectedSeg3']
        assert result['SegmentLabel'].astype(str).equals(expected)

    def test_col_name(self, label_data):
        result = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1,
            col_name='Custom',
        )
        assert 'Custom' in result.columns
        assert 'SegmentLabel' not in result.columns

    def test_step_alias(self, label_data):
        result = col.add_segment_labels(
            label_data.rename(columns={'Step': 'StepAlias'}),
            segment_labels=self.step_labels1,
            step_alias='StepAlias',
        )

        expected = label_data['ExpectedSeg1']
        assert result['SegmentLabel'].astype(str).equals(expected)

    def test_cycle_alias(self, label_data):
        result = col.add_segment_labels(
            label_data.rename(columns={'Cycle': 'CycleAlias'}),
            segment_labels=self.cycle_labels2,
            cycle_alias='CycleAlias',
        )

        expected = label_data['ExpectedSeg2']
        assert result['SegmentLabel'].astype(str).equals(expected)

    def test_custom_default_labels(self, label_data):
        result = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1,
            default='None',
        )

        expected = label_data['ExpectedSeg1'].replace({'Unlabeled': 'None'})

        assert 'None' in expected.to_list()
        assert result['SegmentLabel'].astype(str).equals(expected)

    def test_reset_false_accumulates_labels(self, label_data):
        first = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1[:1],
        )
        second = col.add_segment_labels(
            first,
            segment_labels=self.step_labels1[1:],
            reset=False,
        )

        expected = label_data['ExpectedSeg1']
        assert second['SegmentLabel'].astype(str).equals(expected)

    def test_reset_true_clears_labels(self, label_data):
        first = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1[:1],
        )
        assert 'A' in first['SegmentLabel'].astype(str).tolist()

        second = col.add_segment_labels(
            first,
            segment_labels=self.step_labels1[1:],
            reset=True,
        )
        assert 'A' not in second['SegmentLabel'].astype(str).tolist()

        expected = label_data['ExpectedSeg1'].replace({'A': 'Unlabeled'})
        assert second['SegmentLabel'].astype(str).equals(expected)

    def test_returns_copy(self, label_data):
        result = col.add_segment_labels(
            label_data,
            segment_labels=self.step_labels1,
        )

        assert result is not label_data
        assert 'SegmentLabel' not in label_data.columns

    def test_output_is_categorical(self, label_data):
        labels = [col.SegmentLabel('Early', step_nums=[1])]
        result = col.add_segment_labels(label_data, segment_labels=labels)
        assert isinstance(result['SegmentLabel'].dtype, pd.CategoricalDtype)


class TestAddState:
    def test_missing_amps_column_raises(self, state_data):
        ds = state_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            col.add_state(ds)

    def test_grouped_charge_discharge_rest(self, state_data):
        result = col.add_state(state_data, which='Step')

        expected = state_data['ExpectedStep']
        assert result['State'].astype(str).equals(expected)

    def test_which_none_is_row_by_row(self, state_data):
        result = col.add_state(state_data, which=None)

        expected = state_data['ExpectedNone']
        assert result['State'].astype(str).equals(expected)

    def test_col_name(self, state_data):
        result = col.add_state(state_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'State' not in result.columns

    def test_amps_alias(self, state_data):
        ds = state_data.rename(columns={'Amps': 'Current'})
        result = col.add_state(ds, amps_alias='Current')
        assert 'State' in result.columns

    def test_returns_copy(self, state_data):
        result = col.add_state(state_data)
        assert result is not state_data
        assert 'State' not in state_data.columns

    def test_output_is_categorical(self, state_data):
        result = col.add_state(state_data)
        assert isinstance(result['State'].dtype, pd.CategoricalDtype)


class TestAddControlMode:
    def test_missing_columns_raises(self, control_data):
        ds = control_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            col.add_control_mode(ds)

        ds = control_data.drop(columns=['Volts'])
        with pytest.raises(ValueError):
            col.add_control_mode(ds)

    def test_constant_current_detected(self, control_data):
        result = col.add_control_mode(control_data)

        cc_rows = result[control_data['Expected'] == 'CC']
        assert (cc_rows['ControlMode'].astype(str) == 'CC').all()

    def test_constant_voltage_detected(self, control_data):
        result = col.add_control_mode(control_data)

        cv_rows = result[control_data['Expected'] == 'CV']
        assert (cv_rows['ControlMode'].astype(str) == 'CV').all()

    def test_exact_zero_rests_detected(self, control_data):
        result = col.add_control_mode(control_data)

        rest_rows = result[control_data['Expected'] == 'Rest']
        assert (rest_rows['ControlMode'].astype(str) == 'Rest').all()

    def test_ambiguous_groups_unknown(self, control_data):
        result = col.add_control_mode(control_data)

        # include CP in unknown since watts not given
        unknown_rows = result[control_data['Expected'].isin(['CP', 'Unknown'])]
        assert (unknown_rows['ControlMode'].astype(str) == 'Unknown').all()

    def test_custom_default_mode(self, control_data):
        result = col.add_control_mode(control_data, default='Mixed')

        # include CP in unknown since watts not given
        unknown_rows = result[control_data['Expected'].isin(['CP', 'Unknown'])]
        assert (unknown_rows['ControlMode'].astype(str) == 'Mixed').all()

    def test_constant_power_detected_when_watts_given(self, control_data):
        ds = control_data.copy()
        ds['Watts'] = ds['Amps'] * ds['Volts']

        without_watts = col.add_control_mode(ds, watts_alias=None)
        with_watts = col.add_control_mode(ds, watts_alias='Watts')

        cp_without = without_watts[control_data['Expected'] == 'CP']
        cp_with = with_watts[control_data['Expected'] == 'CP']

        assert (cp_without['ControlMode'] == 'Unknown').all()
        assert (cp_with['ControlMode'] == 'CP').all()

    def test_rtol_dict_missing_key_raises(self, control_data):
        ds = control_data.copy()
        ds['Watts'] = ds['Amps'] * ds['Volts']

        # 'CP' is required since watts_alias is given, but missing from rtol
        with pytest.raises(KeyError, match='.*Missing keys in rtol.*'):
            col.add_control_mode(
                ds,
                watts_alias='Watts',
                rtol={'CC': 5e-3, 'CV': 5e-3},
            )

    def test_rtol_dict_per_mode_tolerance(self, control_data):
        ds = control_data.copy()

        # loose 'CC' tolerance also matches, so both modes match -> Unknown
        expected = control_data['Expected'].replace({'CP': 'Unknown'})
        loose = col.add_control_mode(ds, rtol={'CC': 5e-2, 'CV': 5e-3})
        assert loose['ControlMode'].astype(str).equals(expected)

        # tight 'CC' tolerance excludes that match, leaving only 'CV'
        rng = np.random.default_rng(seed=42)

        cc_rows = ds['Expected'] == 'CC'
        ds.loc[cc_rows, 'Amps'] *= rng.normal(0, 1e-3, sum(cc_rows))

        expected = expected.replace({'CC': 'Unknown'})
        tight = col.add_control_mode(ds, rtol={'CC': 0.0, 'CV': 5e-3})
        assert tight['ControlMode'].astype(str).equals(expected)

    def test_which_alias(self, control_data):
        ds = control_data.rename(columns={'Step': 'Segment'})
        result = col.add_control_mode(ds, which='Segment')

        expected = control_data['Expected'].replace({'CP': 'Unknown'})
        assert result['ControlMode'].astype(str).equals(expected)

    def test_returns_copy(self, control_data):
        result = col.add_control_mode(control_data)
        assert result is not control_data
        assert 'ControlMode' not in control_data.columns

    def test_output_is_categorical(self, control_data):
        result = col.add_control_mode(control_data)
        assert isinstance(result['ControlMode'].dtype, pd.CategoricalDtype)
