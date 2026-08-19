import pytest
import numpy.testing as npt

import ampworks as amp

from ampworks.columns import add_instance_nums, add_relative_time


@pytest.fixture
def sequencing_data():
    """Step 1 repeats within Cycle 1, then again within Cycle 2. The Step
    value itself doesn't change across the cycle boundary (rows 5-6), so
    'fast' (Step-only) grouping merges what 'cycle_resets' keeps separate.
    """
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
        'Cycle': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'Seconds': [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.],
        'ExpectedInstanceNum': [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
        'ExpectedStepTime': [0., 1., 0., 1., 0., 1., 2., 3., 0., 1., 0., 1.],
    })


class TestAddInstanceNums:

    def test_missing_step_column_raises(self, sequencing_data):
        ds = sequencing_data.drop(columns=['Step'])
        with pytest.raises(ValueError):
            add_instance_nums(ds)

    def test_missing_cycle_column_raises_when_cycle_resets(
        self, sequencing_data,
    ):
        ds = sequencing_data.drop(columns=['Cycle'])
        with pytest.raises(ValueError):
            add_instance_nums(ds)

    def test_default_settings(self, sequencing_data):
        result = add_instance_nums(sequencing_data)
        npt.assert_array_equal(
            result['InstanceNum'], sequencing_data['ExpectedInstanceNum'],
        )

    def test_cycle_resets_false_is_global(self, sequencing_data):
        result = add_instance_nums(sequencing_data, cycle_resets=False)
        expected = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
        npt.assert_array_equal(result['InstanceNum'], expected)

    def test_fast_option_warns_and_ignores_cycle_resets(self, sequencing_data):
        with pytest.warns(UserWarning, match='fast=True'):
            result = add_instance_nums(sequencing_data, fast=True)

        expected = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5]
        npt.assert_array_equal(result['InstanceNum'], expected)

    def test_col_name(self, sequencing_data):
        result = add_instance_nums(sequencing_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'InstanceNum' not in result.columns

    def test_returns_copy(self, sequencing_data):
        result = add_instance_nums(sequencing_data)
        assert result is not sequencing_data
        assert 'InstanceNum' not in sequencing_data.columns


class TestAddRelativeTime:

    def test_missing_columns_raises(self, sequencing_data):
        ds = sequencing_data.drop(columns=['Seconds'])
        with pytest.raises(ValueError):
            add_relative_time(ds)

    def test_default_settings(self, sequencing_data):
        result = add_relative_time(sequencing_data)
        npt.assert_allclose(
            result['StepTime'], sequencing_data['ExpectedStepTime'],
        )

    def test_which_alias(self, sequencing_data):
        ds = sequencing_data.rename(columns={'Step': 'Segment'})
        result = add_relative_time(ds, which='Segment')
        npt.assert_allclose(
            result['StepTime'], sequencing_data['ExpectedStepTime'],
        )

    def test_time_alias(self, sequencing_data):
        ds = sequencing_data.rename(columns={'Seconds': 'Elapsed'})
        result = add_relative_time(ds, time_alias='Elapsed')
        npt.assert_allclose(
            result['StepTime'], sequencing_data['ExpectedStepTime'],
        )

    def test_col_name(self, sequencing_data):
        result = add_relative_time(sequencing_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'StepTime' not in result.columns

    def test_returns_copy(self, sequencing_data):
        result = add_relative_time(sequencing_data)
        assert result is not sequencing_data
        assert 'StepTime' not in sequencing_data.columns
