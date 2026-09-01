import pytest
import numpy as np
import numpy.testing as npt

import ampworks as amp
import ampworks.columns as col


@pytest.fixture
def two_step_data():
    """Two steps, each with its own constant power, uniform 1 s spacing."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2],
        'State': ['C', 'C', 'C', 'D', 'D', 'D'],
        'Seconds': [0., 1., 2., 3., 4., 5.],
        'Watts': [1., 1., 1., -2., -2., -2.],
        'ExpectedWh': np.array([0., 1., 2., 0., 2., 4.]) / 3600.,
        'ExpectedCumulativeWh': np.array([0., 1., 2., 1.5, -0.5, -2.5]) / 3600.,
        'ExpectedThroughputWh': np.array([0., 1., 2., 3.5, 5.5, 7.5]) / 3600.,
    })


@pytest.fixture
def diverging_which_data():
    """Step changes mid-state, so `which='Step'`/`'State'` results differ."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2],
        'State': ['C', 'C', 'C', 'C'],
        'Seconds': [0., 1., 2., 3.],
        'Watts': [1., 1., 1., 1.],
        'ExpectedByStep': np.array([0., 1., 0., 1.]) / 3600.,
        'ExpectedByState': np.array([0., 1., 2., 3.]) / 3600.,
    })


class TestAddEnergy:

    def test_default_settings(self, two_step_data):
        result = col.add_energy(two_step_data)
        npt.assert_allclose(result['Wh'], two_step_data['ExpectedWh'])

    def test_which_alias_changes_reset_points(self, diverging_which_data):
        by_step = col.add_energy(diverging_which_data, which='Step')
        by_state = col.add_energy(diverging_which_data, which='State')

        npt.assert_allclose(
            by_step['Wh'], diverging_which_data['ExpectedByStep'],
        )
        npt.assert_allclose(
            by_state['Wh'], diverging_which_data['ExpectedByState'],
        )

    def test_col_name(self, two_step_data):
        result = col.add_energy(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'Wh' not in result.columns

    def test_seconds_alias(self, two_step_data):
        ds = two_step_data.rename(columns={'Seconds': 'Time'})
        result = col.add_energy(ds, seconds_alias='Time')
        npt.assert_allclose(result['Wh'], two_step_data['ExpectedWh'])

    def test_watts_alias(self, two_step_data):
        ds = two_step_data.rename(columns={'Watts': 'Power'})
        result = col.add_energy(ds, watts_alias='Power')
        npt.assert_allclose(result['Wh'], two_step_data['ExpectedWh'])

    def test_missing_columns_raises(self, two_step_data):
        ds = two_step_data.drop(columns=['Watts'])
        with pytest.raises(ValueError):
            col.add_energy(ds)

    def test_returns_copy(self, two_step_data):
        result = col.add_energy(two_step_data)
        assert result is not two_step_data
        assert 'Wh' not in two_step_data.columns


class TestAddCumulativeEnergy:

    def test_integral_method(self, two_step_data):
        result = col.add_cumulative_energy(two_step_data)
        npt.assert_allclose(
            result['CumulativeWh'], two_step_data['ExpectedCumulativeWh'],
        )

    def test_wh_column_method(self, two_step_data):
        # 'wh_column' reconstructs cumulative Wh from discrete increments in an
        # existing Wh column (which already resets to zero each step), so it is
        # not expected to numerically match the 'integral' method at resets
        with_wh = col.add_energy(two_step_data)
        result = col.add_cumulative_energy(with_wh, method='wh_column')
        expected = np.array([0., 1., 2., 2., 0., -2.]) / 3600.
        npt.assert_allclose(result['CumulativeWh'], expected, atol=1e-12)

    def test_invalid_method_raises(self, two_step_data):
        with pytest.raises(ValueError):
            col.add_cumulative_energy(two_step_data, method='bogus')

    def test_col_name(self, two_step_data):
        result = col.add_cumulative_energy(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'CumulativeWh' not in result.columns

    def test_returns_copy(self, two_step_data):
        result = col.add_cumulative_energy(two_step_data)
        assert result is not two_step_data
        assert 'CumulativeWh' not in two_step_data.columns


class TestAddThroughputEnergy:

    def test_integral_method(self, two_step_data):
        result = col.add_throughput_energy(two_step_data)
        npt.assert_allclose(
            result['ThroughputWh'], two_step_data['ExpectedThroughputWh'],
        )

    def test_wh_column_method(self, two_step_data):
        # 'wh_column' reconstructs throughput Wh from discrete increments in an
        # existing Wh column (which already resets to zero each step), so it is
        # not expected to numerically match the 'integral' method at resets
        with_wh = col.add_energy(two_step_data)
        result = col.add_throughput_energy(with_wh, method='wh_column')
        expected = np.array([0., 1., 2., 2., 4., 6.]) / 3600.
        npt.assert_allclose(result['ThroughputWh'], expected, atol=1e-12)

    def test_invalid_method_raises(self, two_step_data):
        with pytest.raises(ValueError):
            col.add_throughput_energy(two_step_data, method='bogus')

    def test_col_name(self, two_step_data):
        result = col.add_throughput_energy(two_step_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'ThroughputWh' not in result.columns

    def test_returns_copy(self, two_step_data):
        result = col.add_throughput_energy(two_step_data)
        assert result is not two_step_data
        assert 'ThroughputWh' not in two_step_data.columns
