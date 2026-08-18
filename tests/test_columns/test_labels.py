import pytest
import pandas as pd

import ampworks as amp
import ampworks.columns as awc


@pytest.fixture
def step_data():
    """Three steps (1, 2, 3), two rows each, across two cycles."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 3, 3],
        'Cycle': [1, 1, 1, 1, 2, 2],
    })


@pytest.fixture
def state_data():
    """Steps demonstrating charge, discharge, rest, taper, and mixed sign."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
        'Amps': [
            1., 1., 1.,     # all positive -> charge
            -1., -1., -1.,  # all negative -> discharge
            0., 0., 0.,     # exactly zero -> rest
            1., 0.5, 0.,    # tapers to zero, stays >= 0 -> charge
            -1., 0., 1.,    # truly mixed sign -> unknown
        ],
    })


@pytest.fixture
def control_mode_data():
    """Steps demonstrating CC, CV, rest, an ambiguous case, and CP."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7],
        'Amps': [
            1.0, 1.0, 1.0,     # constant current (charge) -> CC
            -1.0, -1.0, -1.0,  # constant current (discharge) -> CC
            1.0, 0.5, 0.1,     # varying current, constant voltage -> CV
            0.0, 0.0, 0.0,     # exact rest -> Rest (overrides any match)
            1.0, 2.0, 3.0,     # varying current and voltage -> Unknown
            1.0, 2.0, 4.0,     # varying, but constant power -> CP
            1.0, 1.0, 1.0,     # ambiguous, meets CC and CV -> Unknown
        ],
        'Volts': [
            3.0, 3.5, 4.0,
            4.0, 3.5, 3.0,
            4.2, 4.2, 4.2,
            4.15, 4.16, 4.17,
            3.0, 3.5, 4.0,
            4.0, 2.0, 1.0,
            4.2, 4.2, 4.2,
        ],
    })


class TestStepLabel:

    def test_type_errors(self):
        with pytest.raises(TypeError):  # label should be str
            awc.StepLabel(label=1, step_num=1)

        with pytest.raises(TypeError):  # step_num should be int
            awc.StepLabel(label='Rest', step_num='1')

    def test_init(self):
        label = awc.StepLabel('Rest', 1)
        assert label.label == 'Rest'
        assert label.step_num == 1

    def test_repr(self):
        label = awc.StepLabel('Rest', 1)
        assert repr(label) == 'StepLabel(label=Rest, step_num=1)'


class TestSegmentLabel:

    def test_requires_exactly_one_of_step_or_cycle_nums(self):
        with pytest.raises(ValueError):  # neither given
            awc.SegmentLabel('HPPC')

        with pytest.raises(ValueError):  # both given
            awc.SegmentLabel('HPPC', step_nums=[1], cycle_nums=[1])

    def test_step_nums_type_errors(self):
        with pytest.raises(TypeError):  # not a Sequence
            awc.SegmentLabel('HPPC', step_nums=1)

        with pytest.raises(TypeError):  # not int items
            awc.SegmentLabel('HPPC', step_nums=['1', '2'])

    def test_cycle_nums_type_errors(self):
        with pytest.raises(TypeError):  # not a Sequence
            awc.SegmentLabel('RPT', cycle_nums=1)

        with pytest.raises(TypeError):  # not int items
            awc.SegmentLabel('RPT', cycle_nums=['1', '2'])

    def test_init_with_step_nums(self):
        label = awc.SegmentLabel('HPPC', step_nums=[4, 5, 6])
        assert label.label == 'HPPC'
        assert label.step_nums == [4, 5, 6]
        assert label.cycle_nums is None

    def test_init_with_cycle_nums(self):
        label = awc.SegmentLabel('RPT', cycle_nums=[1, 2])
        assert label.label == 'RPT'
        assert label.cycle_nums == [1, 2]
        assert label.step_nums is None

    def test_repr(self):
        step_label = awc.SegmentLabel('HPPC', step_nums=[1, 2])
        cycle_label = awc.SegmentLabel('RPT', cycle_nums=[1, 2])

        assert repr(step_label) == 'SegmentLabel(label=HPPC, step_nums=[1, 2])'
        assert repr(cycle_label) == 'SegmentLabel(label=RPT, cycle_nums=[1, 2])'


class TestAddStepLabels:

    labels = [
        awc.StepLabel('A', 1),
        awc.StepLabel('B', 2),
    ]

    def test_keyword_only(self, step_data):
        with pytest.raises(TypeError, match='positional'):
            awc.add_step_labels(step_data, self.labels)

    def test_missing_step_column_raises(self, step_data):
        ds = step_data.rename(columns={'Step': 'StepAlias'})
        with pytest.raises(ValueError):
            awc.add_step_labels(ds, step_labels=self.labels)

    def test_step_labels_must_be_step_label_instances(self, step_data):
        with pytest.raises(TypeError):
            awc.add_step_labels(step_data, step_labels=[('A', 1)])

    def test_basic_labeling_and_default(self, step_data):
        result = awc.add_step_labels(step_data, step_labels=self.labels)

        expected = ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled']
        assert result['StepLabel'].astype(str).tolist() == expected

    def test_returns_copy(self, step_data):
        result = awc.add_step_labels(step_data, step_labels=self.labels)
        assert result is not step_data
        assert 'StepLabel' not in step_data.columns

    def test_col_name(self, step_data):
        result = awc.add_step_labels(
            step_data, step_labels=self.labels, col_name='Custom',
        )
        assert 'Custom' in result.columns
        assert 'StepLabel' not in result.columns

    def test_step_alias(self, step_data):
        ds = step_data.rename(columns={'Step': 'StepAlias'})
        result = awc.add_step_labels(
            ds, step_labels=self.labels, step_alias='StepAlias',
        )

        expected = ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled']
        assert result['StepLabel'].astype(str).tolist() == expected

    def test_default(self, step_data):
        result = awc.add_step_labels(
            step_data, step_labels=self.labels, default='None',
        )
        assert result['StepLabel'].astype(str).tolist()[-2:] == ['None', 'None']

    def test_reset_false_accumulates_across_calls(self, step_data):
        first = awc.add_step_labels(
            step_data, step_labels=[awc.StepLabel('A', 1)],
        )
        second = awc.add_step_labels(
            first, step_labels=[awc.StepLabel('B', 2)], reset=False,
        )

        expected = ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled']
        assert second['StepLabel'].astype(str).tolist() == expected

    def test_reset_true_clears_previous_labels(self, step_data):
        first = awc.add_step_labels(
            step_data, step_labels=[awc.StepLabel('A', 1)],
        )
        second = awc.add_step_labels(
            first, step_labels=[awc.StepLabel('B', 2)], reset=True,
        )

        expected = [
            'Unlabeled', 'Unlabeled', 'B', 'B', 'Unlabeled', 'Unlabeled',
        ]
        assert second['StepLabel'].astype(str).tolist() == expected

    def test_output_is_categorical(self, step_data):
        result = awc.add_step_labels(step_data, step_labels=self.labels)
        assert isinstance(result['StepLabel'].dtype, pd.CategoricalDtype)


class TestAddSegmentLabels:

    def test_keyword_only(self, step_data):
        labels = [awc.SegmentLabel('X', step_nums=[1])]
        with pytest.raises(TypeError, match='positional'):
            awc.add_segment_labels(step_data, labels)

    def test_segment_labels_must_be_segment_label_instances(self, step_data):
        with pytest.raises(TypeError):
            awc.add_segment_labels(step_data, segment_labels=[('X', [1])])

    def test_step_nums_labeling(self, step_data):
        labels = [awc.SegmentLabel('Early', step_nums=[1, 2])]
        result = awc.add_segment_labels(step_data, segment_labels=labels)

        expected = ['Early'] * 4 + ['Unlabeled', 'Unlabeled']
        assert result['SegmentLabel'].astype(str).tolist() == expected

    def test_cycle_nums_labeling(self, step_data):
        labels = [awc.SegmentLabel('Second', cycle_nums=[2])]
        result = awc.add_segment_labels(step_data, segment_labels=labels)

        expected = ['Unlabeled'] * 4 + ['Second', 'Second']
        assert result['SegmentLabel'].astype(str).tolist() == expected

    def test_mixed_step_and_cycle_labels(self, step_data):
        labels = [
            awc.SegmentLabel('Early', step_nums=[1]),
            awc.SegmentLabel('Second', cycle_nums=[2]),
        ]
        result = awc.add_segment_labels(step_data, segment_labels=labels)

        expected = [
            'Early', 'Early', 'Unlabeled', 'Unlabeled', 'Second', 'Second',
        ]
        assert result['SegmentLabel'].astype(str).tolist() == expected

    def test_cycle_only_labels_do_not_require_step_alias(self, step_data):
        ds = step_data.drop(columns=['Step'])
        labels = [awc.SegmentLabel('Second', cycle_nums=[2])]

        result = awc.add_segment_labels(ds, segment_labels=labels)
        assert 'SegmentLabel' in result.columns

    def test_step_only_labels_do_not_require_cycle_alias(self, step_data):
        ds = step_data.drop(columns=['Cycle'])
        labels = [awc.SegmentLabel('Early', step_nums=[1])]

        result = awc.add_segment_labels(ds, segment_labels=labels)
        assert 'SegmentLabel' in result.columns

    def test_reset_false_accumulates_across_calls(self, step_data):
        first = awc.add_segment_labels(
            step_data,
            segment_labels=[awc.SegmentLabel('A', step_nums=[1])],
        )
        second = awc.add_segment_labels(
            first,
            segment_labels=[awc.SegmentLabel('B', step_nums=[2])],
            reset=False,
        )

        expected = ['A', 'A', 'B', 'B', 'Unlabeled', 'Unlabeled']
        assert second['SegmentLabel'].astype(str).tolist() == expected

    def test_reset_true_clears_previous_labels(self, step_data):
        first = awc.add_segment_labels(
            step_data,
            segment_labels=[awc.SegmentLabel('A', step_nums=[1])],
        )
        second = awc.add_segment_labels(
            first,
            segment_labels=[awc.SegmentLabel('B', step_nums=[2])],
            reset=True,
        )

        expected = [
            'Unlabeled', 'Unlabeled', 'B', 'B', 'Unlabeled', 'Unlabeled',
        ]
        assert second['SegmentLabel'].astype(str).tolist() == expected

    def test_output_is_categorical(self, step_data):
        labels = [awc.SegmentLabel('Early', step_nums=[1])]
        result = awc.add_segment_labels(step_data, segment_labels=labels)
        assert isinstance(result['SegmentLabel'].dtype, pd.CategoricalDtype)

    def test_returns_copy(self, step_data):
        labels = [awc.SegmentLabel('Early', step_nums=[1])]
        result = awc.add_segment_labels(step_data, segment_labels=labels)
        assert result is not step_data
        assert 'SegmentLabel' not in step_data.columns


class TestAddState:

    def test_missing_amps_column_raises(self, state_data):
        ds = state_data.rename(columns={'Amps': 'Current'})
        with pytest.raises(ValueError):
            awc.add_state(ds)

    def test_which_none_is_row_by_row(self):
        data = amp.Dataset({'Amps': [2., -3., 0., 0.5, -0.5]})
        result = awc.add_state(data, which=None)

        expected = ['C', 'D', 'R', 'C', 'D']
        assert result['State'].astype(str).tolist() == expected

    def test_grouped_charge_discharge_rest(self, state_data):
        result = awc.add_state(state_data)

        expected = (
            ['C'] * 3 +        # step 1: all positive
            ['D'] * 3 +        # step 2: all negative
            ['R'] * 3 +        # step 3: all exactly zero
            ['C'] * 3 +        # step 4: tapers to zero, stays >= 0
            ['Unknown'] * 3    # step 5: truly mixed sign
        )
        assert result['State'].astype(str).tolist() == expected

    def test_col_name(self, state_data):
        result = awc.add_state(state_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'State' not in result.columns

    def test_amps_alias(self, state_data):
        ds = state_data.rename(columns={'Amps': 'Current'})
        result = awc.add_state(ds, amps_alias='Current')
        assert 'State' in result.columns

    def test_output_is_categorical(self, state_data):
        result = awc.add_state(state_data)
        assert isinstance(result['State'].dtype, pd.CategoricalDtype)

    def test_returns_copy(self, state_data):
        result = awc.add_state(state_data)
        assert result is not state_data
        assert 'State' not in state_data.columns


class TestAddControlMode:

    def test_missing_columns_raises(self, control_mode_data):
        ds = control_mode_data.drop(columns=['Volts'])
        with pytest.raises(ValueError):
            awc.add_control_mode(ds)

    def test_constant_current_detected_as_cc(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)

        step1 = result[result['Step'] == 1]
        step2 = result[result['Step'] == 2]
        assert (step1['ControlMode'] == 'CC').all()
        assert (step2['ControlMode'] == 'CC').all()

    def test_constant_voltage_detected_as_cv(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)

        step3 = result[result['Step'] == 3]
        assert (step3['ControlMode'] == 'CV').all()

    def test_exact_rest_overrides_any_match(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)

        step4 = result[result['Step'] == 4]
        assert (step4['ControlMode'] == 'Rest').all()

    def test_ambiguous_group_uses_default(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)

        step5 = result[result['Step'].isin([5, 7])]
        assert (step5['ControlMode'] == 'Unknown').all()

    def test_custom_default(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data, default='Mixed')

        step5 = result[result['Step'].isin([5, 7])]
        assert (step5['ControlMode'] == 'Mixed').all()

    def test_cp_requires_watts_alias(self, control_mode_data):
        ds = control_mode_data.copy()
        ds['Watts'] = ds['Amps'] * ds['Volts']

        without_watts = awc.add_control_mode(ds)
        with_watts = awc.add_control_mode(ds, watts_alias='Watts')

        step6_without = without_watts[without_watts['Step'] == 6]
        step6_with = with_watts[with_watts['Step'] == 6]

        assert (step6_without['ControlMode'] == 'Unknown').all()
        assert (step6_with['ControlMode'] == 'CP').all()

    def test_which_alias(self, control_mode_data):
        ds = control_mode_data.rename(columns={'Step': 'Segment'})
        result = awc.add_control_mode(ds, which='Segment')

        assert (result[result['Segment'] == 1]['ControlMode'] == 'CC').all()

    def test_output_is_categorical(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)
        assert isinstance(result['ControlMode'].dtype, pd.CategoricalDtype)

    def test_returns_copy(self, control_mode_data):
        result = awc.add_control_mode(control_mode_data)
        assert result is not control_mode_data
        assert 'ControlMode' not in control_mode_data.columns
