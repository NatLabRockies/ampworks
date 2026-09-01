import pytest
import numpy as np
import numpy.testing as npt

import ampworks as amp
import ampworks.columns as col


# fmt: off

@pytest.fixture
def two_step_data():
    """Two steps, each with its own constant current, uniform 1 s spacing."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2],
        'State': ['C', 'C', 'C', 'D', 'D', 'D'],
        'Seconds': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        'Amps': [1.0, 1.0, 1.0, -2.0, -2.0, -2.0],
        'ExpectedAh': np.array([0.0, 1.0, 2.0, 0.0, 2.0, 4.0]) / 3600.0,
        'ExpectedCumulativeAh':
            np.array([0.0, 1.0, 2.0, 1.5, -0.5, -2.5]) / 3600.0,
        'ExpectedThroughputAh':
            np.array([0.0, 1.0, 2.0, 3.5, 5.5, 7.5]) / 3600.0,
    })


@pytest.fixture
def diverging_which_data():
    """Step changes mid-state, so `which='Step'`/`'State'` results differ."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2],
        'State': ['C', 'C', 'C', 'C'],
        'Seconds': [0.0, 1.0, 2.0, 3.0],
        'Amps': [1.0, 1.0, 1.0, 1.0],
        'ExpectedByStep': np.array([0.0, 1.0, 0.0, 1.0]) / 3600.0,
        'ExpectedByState': np.array([0.0, 1.0, 2.0, 3.0]) / 3600.0,
    })


@pytest.fixture
def throughput_data():
    return amp.Dataset({'ThroughputAh': [0.0, 1.0, 2.0, 4.0]})

# fmt: on


class TestAddCapacity:
    def test_default_settings(self, two_step_data):
        result = col.add_capacity(two_step_data)
        npt.assert_allclose(result['Ah'], two_step_data['ExpectedAh'])

    def test_which_alias_changes_reset_points(self, diverging_which_data):
        by_step = col.add_capacity(diverging_which_data, which='Step')
        by_state = col.add_capacity(diverging_which_data, which='State')

        npt.assert_allclose(
            by_step['Ah'],
            diverging_which_data['ExpectedByStep'],
        )
        npt.assert_allclose(
            by_state['Ah'],
            diverging_which_data['ExpectedByState'],
        )

    def test_col_name(self, two_step_data):
        result = col.add_capacity(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'Ah' not in result.columns

    def test_seconds_alias(self, two_step_data):
        ds = two_step_data.rename(columns={'Seconds': 'Time'})
        result = col.add_capacity(ds, seconds_alias='Time')
        npt.assert_allclose(result['Ah'], two_step_data['ExpectedAh'])

    def test_amps_alias(self, two_step_data):
        ds = two_step_data.rename(columns={'Amps': 'Current'})
        result = col.add_capacity(ds, amps_alias='Current')
        npt.assert_allclose(result['Ah'], two_step_data['ExpectedAh'])

    def test_missing_columns_raises(self, two_step_data):
        ds = two_step_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            col.add_capacity(ds)

    def test_returns_copy(self, two_step_data):
        result = col.add_capacity(two_step_data)
        assert result is not two_step_data
        assert 'Ah' not in two_step_data.columns


class TestAddCumulativeCapacity:
    def test_integral_method(self, two_step_data):
        result = col.add_cumulative_capacity(two_step_data)
        npt.assert_allclose(
            result['CumulativeAh'],
            two_step_data['ExpectedCumulativeAh'],
        )

    def test_ah_column_method(self, two_step_data):
        # 'ah_column' reconstructs cumulative Ah from discrete increments in an
        # existing Ah column (which already resets to zero each step), so it is
        # not expected to numerically match the 'integral' method at resets
        with_ah = col.add_capacity(two_step_data)
        result = col.add_cumulative_capacity(with_ah, method='ah_column')
        expected = np.array([0.0, 1.0, 2.0, 2.0, 0.0, -2.0]) / 3600.0
        npt.assert_allclose(result['CumulativeAh'], expected, atol=1e-12)

    def test_invalid_method_raises(self, two_step_data):
        with pytest.raises(ValueError):
            col.add_cumulative_capacity(two_step_data, method='bogus')

    def test_col_name(self, two_step_data):
        result = col.add_cumulative_capacity(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'CumulativeAh' not in result.columns

    def test_returns_copy(self, two_step_data):
        result = col.add_cumulative_capacity(two_step_data)
        assert result is not two_step_data
        assert 'CumulativeAh' not in two_step_data.columns


class TestAddThroughputCapacity:
    def test_integral_method(self, two_step_data):
        result = col.add_throughput_capacity(two_step_data)
        npt.assert_allclose(
            result['ThroughputAh'],
            two_step_data['ExpectedThroughputAh'],
        )

    def test_ah_column_method(self, two_step_data):
        # 'ah_column' reconstructs throughput Ah from discrete increments in an
        # existing Ah column (which already resets to zero each step), so it is
        # not expected to numerically match the 'integral' method at resets
        with_ah = col.add_capacity(two_step_data)
        result = col.add_throughput_capacity(with_ah, method='ah_column')
        expected = np.array([0.0, 1.0, 2.0, 2.0, 4.0, 6.0]) / 3600.0
        npt.assert_allclose(result['ThroughputAh'], expected, atol=1e-12)

    def test_invalid_method_raises(self, two_step_data):
        with pytest.raises(ValueError):
            col.add_throughput_capacity(two_step_data, method='bogus')

    def test_col_name(self, two_step_data):
        result = col.add_throughput_capacity(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'ThroughputAh' not in result.columns

    def test_returns_copy(self, two_step_data):
        result = col.add_throughput_capacity(two_step_data)
        assert result is not two_step_data
        assert 'ThroughputAh' not in two_step_data.columns


class TestAddEquivalentFullCycles:
    def test_default_settings(self, throughput_data):
        result = col.add_equivalent_full_cycles(throughput_data, nominal_ah=2.0)
        npt.assert_allclose(result['EFC'], [0.0, 0.25, 0.5, 1.0])

    def test_negative_nominal_ah_uses_magnitude(self, throughput_data):
        result = col.add_equivalent_full_cycles(
            throughput_data,
            nominal_ah=-2.0,
        )
        npt.assert_allclose(result['EFC'], [0.0, 0.25, 0.5, 1.0])

    def test_zero_nominal_ah_raises(self, throughput_data):
        with pytest.raises(ValueError):
            col.add_equivalent_full_cycles(throughput_data, nominal_ah=0.0)

    def test_missing_column_raises(self, throughput_data):
        ds = throughput_data.drop(columns=['ThroughputAh'])
        with pytest.raises(ValueError):
            col.add_equivalent_full_cycles(ds, nominal_ah=2.0)

    def test_col_name(self, throughput_data):
        result = col.add_equivalent_full_cycles(
            throughput_data,
            nominal_ah=2.0,
            col_name='Custom',
        )
        assert 'Custom' in result.columns
        assert 'EFC' not in result.columns

    def test_throughput_ah_alias(self, throughput_data):
        ds = throughput_data.rename(columns={'ThroughputAh': 'Custom'})
        result = col.add_equivalent_full_cycles(
            ds,
            nominal_ah=2.0,
            throughput_ah_alias='Custom',
        )
        npt.assert_allclose(result['EFC'], [0.0, 0.25, 0.5, 1.0])

    def test_returns_copy(self, throughput_data):
        result = col.add_equivalent_full_cycles(throughput_data, nominal_ah=2.0)
        assert result is not throughput_data
        assert 'EFC' not in throughput_data.columns
