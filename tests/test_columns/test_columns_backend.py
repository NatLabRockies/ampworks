import pytest
import numpy as np
import numpy.testing as npt

import ampworks as amp

from ampworks.columns._backend import (
    _instance_nums,
    _ah_wh,
    _ah_wh_cumulative,
    _ah_wh_throughput,
)


@pytest.fixture
def two_cycles():
    """The repeated_steps pattern repeated a second time in a new cycle."""
    return amp.Dataset(
        {
            'Step': [1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
            'Cycle': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            'ExpectedFast': [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5],
            'ExpectedCycle': [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
            'ExpectedGlobal': [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
        }
    )


@pytest.fixture
def two_step_ahwh():
    """Two steps, each with its own constant current, uniform 1 s spacing."""
    return amp.Dataset(
        {
            'Step': [1, 1, 1, 2, 2, 2],
            'Seconds': [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            'Amps': [1.0, 1.0, 1.0, -2.0, -2.0, -2.0],
            'Expected': np.array([0.0, 1.0, 2.0, 0.0, 2.0, 4.0]) / 3600.0,
            'ExpCumulative': np.array([0.0, 1.0, 2.0, 1.5, -0.5, -2.5])
            / 3600.0,
            'ExpThroughput': np.array([0.0, 1.0, 2.0, 3.5, 5.5, 7.5]) / 3600.0,
        }
    )


REST_STEP = amp.Dataset(
    {  # includes a rest step between active segments
        'Ah': [0.0, 1.0, 3.0, 7.0, 0.0, 0.0, 0.0, 0.0, 2.0, 3.0],
        'State1': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'C', 'C', 'C'],
        'State2': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'D', 'D', 'D'],
        'State3': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'C', 'C', 'C'],
        'State4': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'D', 'D'],
        'CumulativeAh1': [0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 9.0, 10.0],
        'CumulativeAh2': [0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 5.0, 4.0],
        'CumulativeAh3': [
            0.0,
            -1.0,
            -3.0,
            -7.0,
            -7.0,
            -7.0,
            -7.0,
            -7.0,
            -5.0,
            -4.0,
        ],
        'CumulativeAh4': [
            0.0,
            -1.0,
            -3.0,
            -7.0,
            -7.0,
            -7.0,
            -7.0,
            -7.0,
            -9.0,
            -10.0,
        ],
        'ThroughputAh': [0.0, 1.0, 3.0, 7.0, 7.0, 7.0, 7.0, 7.0, 9.0, 10.0],
    }
)

NO_REST = amp.Dataset(
    {  # goes directly from one active step to another
        'Ah': [0.0, 1.0, 3.0, 7.0, 0.0, 2.0, 3.0],
        'State1': ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
        'State2': ['C', 'C', 'C', 'C', 'D', 'D', 'D'],
        'State3': ['D', 'D', 'D', 'D', 'C', 'C', 'C'],
        'State4': ['D', 'D', 'D', 'D', 'D', 'D', 'D'],
        'CumulativeAh1': [0.0, 1.0, 3.0, 7.0, 7.0, 9.0, 10.0],
        'CumulativeAh2': [0.0, 1.0, 3.0, 7.0, 7.0, 5.0, 4.0],
        'CumulativeAh3': [0.0, -1.0, -3.0, -7.0, -7.0, -5.0, -4.0],
        'CumulativeAh4': [0.0, -1.0, -3.0, -7.0, -7.0, -9.0, -10.0],
        'ThroughputAh': [0.0, 1.0, 3.0, 7.0, 7.0, 9.0, 10.0],
    }
)


class TestInstanceNums:
    def test_fast_option(self, two_cycles):
        # warn when requesting both fast and cycle_resets
        with pytest.warns(UserWarning, match='fast=True'):
            result = _instance_nums(
                two_cycles,
                by='Step',
                cycle_col=None,
                cycle_resets=True,
                fast=True,
            )

        assert result.equals(two_cycles['ExpectedFast'])

    def test_cycle_resets_requires_cycle_col(self, two_cycles):
        with pytest.raises(ValueError, match='cycle_col'):
            _instance_nums(
                two_cycles,
                by='Step',
                cycle_col=None,
                cycle_resets=True,
                fast=False,
            )

    def test_cycle_resets_true(self, two_cycles):
        result = _instance_nums(
            two_cycles,
            by='Step',
            cycle_col='Cycle',
            cycle_resets=True,
            fast=False,
        )

        assert result.equals(two_cycles['ExpectedCycle'])

    def test_cycle_resets_false_is_global(self, two_cycles):
        result = _instance_nums(
            two_cycles,
            by='Step',
            cycle_col=None,
            cycle_resets=False,
            fast=False,
        )

        assert result.equals(two_cycles['ExpectedGlobal'])


class TestAhWh:
    def test_missing_columns_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='Missing'):
            _ah_wh(
                two_step_ahwh,
                by='Step',
                seconds_col='Missing',
                value_col='Amps',
            )

    def test_using_amps_for_ah(self, two_step_ahwh):
        result = _ah_wh(
            two_step_ahwh,
            by='Step',
            seconds_col='Seconds',
            value_col='Amps',
        )

        npt.assert_allclose(result.to_numpy(), two_step_ahwh['Expected'])

    def test_using_watts_for_wh(self, two_step_ahwh):
        two_step_ahwh['Watts'] = 10 * two_step_ahwh['Amps']

        result = _ah_wh(
            two_step_ahwh,
            by='Step',
            seconds_col='Seconds',
            value_col='Watts',
        )

        npt.assert_allclose(result.to_numpy(), 10 * two_step_ahwh['Expected'])


class TestAhWhCumulative:
    def test_invalid_method_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_cumulative(
                two_step_ahwh,
                method='bogus',
                seconds_col='Seconds',
                value_col='Amps',
                state_col='State',
                valueh_col='Ah',
            )

    def test_integral_method_against_expected(self, two_step_ahwh):
        result = _ah_wh_cumulative(
            two_step_ahwh,
            method='integral',
            seconds_col='Seconds',
            value_col='Amps',
            state_col='State',
            valueh_col='Ah',
        )

        npt.assert_allclose(result.to_numpy(), two_step_ahwh['ExpCumulative'])

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    def test_column_method_against_expected(self, data):

        for n in range(1, 5):
            result = _ah_wh_cumulative(
                data,
                method='column',
                seconds_col='Seconds',
                value_col='Amps',
                state_col=f'State{n}',
                valueh_col='Ah',
            )

            expected = data[f"CumulativeAh{n}"].to_numpy()
            npt.assert_allclose(result.to_numpy(), expected)

    def test_column_method_negative_ah_raises(self):
        data = amp.Dataset({'Ah': [0.0, -1.0, 2.0], 'State': ['C', 'C', 'C']})
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_cumulative(
                data,
                method='column',
                seconds_col='Seconds',
                value_col='Amps',
                state_col='State',
                valueh_col='Ah',
            )

    def test_column_method_invalid_state_raises(self):
        data = amp.Dataset({'Ah': [0.0, 1.0, 2.0], 'State': ['C', 'X', 'C']})
        with pytest.raises(ValueError, match='State'):
            _ah_wh_cumulative(
                data,
                method='column',
                seconds_col='Seconds',
                value_col='Amps',
                state_col='State',
                valueh_col='Ah',
            )


class TestAhWhThroughput:
    def test_invalid_method_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_throughput(
                two_step_ahwh,
                method='bogus',
                seconds_col='Seconds',
                value_col='Amps',
                valueh_col='Ah',
            )

    def test_integral_uses_absolute_value(self, two_step_ahwh):
        result = _ah_wh_throughput(
            two_step_ahwh,
            method='integral',
            seconds_col='Seconds',
            value_col='Amps',
            valueh_col='Ah',
        )

        # throughput must be monotonically non-decreasing, regardless of sign
        values = result.to_numpy()
        assert np.all(np.diff(values) >= 0)
        assert values[-1] > 0

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    def test_column_matches_hand_verified_values(self, data):
        result = _ah_wh_throughput(
            data,
            method='column',
            seconds_col='Seconds',
            value_col='Amps',
            valueh_col='Ah',
        )

        expected = data['ThroughputAh'].to_numpy()
        npt.assert_allclose(result.to_numpy(), expected)

    def test_column_negative_value_raises(self):
        data = amp.Dataset({'Ah': [0.0, -1.0, 2.0]})
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_throughput(
                data,
                method='column',
                seconds_col='Seconds',
                value_col='Amps',
                valueh_col='Ah',
            )
