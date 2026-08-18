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
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
        'Cycle': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'ExpectedFast': [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5],
        'ExpectedCycle': [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2],
        'ExpectedGlobal': [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3],
    })


@pytest.fixture
def two_step_ahwh():
    """Two steps, each with its own constant current, uniform 1 s spacing."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2],
        'Seconds': [0., 1., 2., 3., 4., 5.],
        'Amps': [1., 1., 1., -2., -2., -2.],
        'Expected': np.array([0., 1., 2., 0., 2., 4.]) / 3600.,
        'ExpCumulative': np.array([0., 1., 2., 1.5, -0.5, -2.5]) / 3600.,
        'ExpThroughput': np.array([0., 1., 2., 3.5, 5.5, 7.5]) / 3600.,
    })


REST_STEP = amp.Dataset({  # includes a rest step between active segments
    'Ah': [0., 1., 3., 7., 0., 0., 0., 0., 2., 3.],
    'State1': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'C', 'C', 'C'],
    'State2': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'D', 'D', 'D'],
    'State3': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'C', 'C', 'C'],
    'State4': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'D', 'D'],
    'CumulativeAh1': [0., 1., 3., 7., 7., 7., 7., 7., 9., 10.],
    'CumulativeAh2': [0., 1., 3., 7., 7., 7., 7., 7., 5., 4.],
    'CumulativeAh3': [0., -1., -3., -7., -7., -7., -7., -7., -5., -4.],
    'CumulativeAh4': [0., -1., -3., -7., -7., -7., -7., -7., -9., -10.],
    'ThroughputAh': [0., 1., 3., 7., 7., 7., 7., 7., 9., 10.],
})

NO_REST = amp.Dataset({  # goes directly from one active step to another
    'Ah': [0., 1., 3., 7., 0., 2., 3.],
    'State1': ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
    'State2': ['C', 'C', 'C', 'C', 'D', 'D', 'D'],
    'State3': ['D', 'D', 'D', 'D', 'C', 'C', 'C'],
    'State4': ['D', 'D', 'D', 'D', 'D', 'D', 'D'],
    'CumulativeAh1': [0., 1., 3., 7., 7., 9., 10.],
    'CumulativeAh2': [0., 1., 3., 7., 7., 5., 4.],
    'CumulativeAh3': [0., -1., -3., -7., -7., -5., -4.],
    'CumulativeAh4': [0., -1., -3., -7., -7., -9., -10.],
    'ThroughputAh': [0., 1., 3., 7., 7., 9., 10.],
})


class TestInstanceNums:

    def test_fast_option(self, two_cycles):
        # warn when requesting both fast and cycle_resets
        with pytest.warns(UserWarning, match='fast=True'):
            result = _instance_nums(
                two_cycles,
                which='Step',
                cycle_alias=None,
                cycle_resets=True,
                fast=True,
            )

        assert result.equals(two_cycles['ExpectedFast'])

    def test_cycle_resets_requires_cycle_alias(self, two_cycles):
        with pytest.raises(ValueError, match='cycle_alias'):
            _instance_nums(
                two_cycles,
                which='Step',
                cycle_alias=None,
                cycle_resets=True,
                fast=False,
            )

    def test_cycle_resets_true(self, two_cycles):
        result = _instance_nums(
            two_cycles,
            which='Step',
            cycle_alias='Cycle',
            cycle_resets=True,
            fast=False,
        )

        assert result.equals(two_cycles['ExpectedCycle'])

    def test_cycle_resets_false_is_global(self, two_cycles):
        result = _instance_nums(
            two_cycles,
            which='Step',
            cycle_alias=None,
            cycle_resets=False,
            fast=False,
        )

        assert result.equals(two_cycles['ExpectedGlobal'])


class TestAhWh:

    def test_missing_columns_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='Missing'):
            _ah_wh(
                two_step_ahwh,
                which='Step',
                seconds_alias='Missing',
                value_alias='Amps',
            )

    def test_using_amps_for_ah(self, two_step_ahwh):
        result = _ah_wh(
            two_step_ahwh,
            which='Step',
            seconds_alias='Seconds',
            value_alias='Amps',
        )

        npt.assert_allclose(result.to_numpy(), two_step_ahwh['Expected'])

    def test_using_watts_for_wh(self, two_step_ahwh):
        two_step_ahwh['Watts'] = 10 * two_step_ahwh['Amps']

        result = _ah_wh(
            two_step_ahwh,
            which='Step',
            seconds_alias='Seconds',
            value_alias='Watts',
        )

        npt.assert_allclose(result.to_numpy(), 10 * two_step_ahwh['Expected'])


class TestAhWhCumulative:

    def test_invalid_method_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_cumulative(
                two_step_ahwh,
                method='bogus',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )

    def test_integral_method_against_expected(self, two_step_ahwh):
        result = _ah_wh_cumulative(
            two_step_ahwh,
            method='integral',
            seconds_alias='Seconds',
            value_alias='Amps',
            state_alias='State',
            valueh_alias='Ah',
        )

        npt.assert_allclose(result.to_numpy(), two_step_ahwh['ExpCumulative'])

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    def test_column_method_against_expected(self, data):

        for n in range(1, 5):
            result = _ah_wh_cumulative(
                data,
                method='column',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias=f'State{n}',
                valueh_alias='Ah',
            )

            expected = data[f"CumulativeAh{n}"].to_numpy()
            npt.assert_allclose(result.to_numpy(), expected)

    def test_column_method_negative_ah_raises(self):
        data = amp.Dataset({'Ah': [0., -1., 2.], 'State': ['C', 'C', 'C']})
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_cumulative(
                data,
                method='column',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )

    def test_column_method_invalid_state_raises(self):
        data = amp.Dataset({'Ah': [0., 1., 2.], 'State': ['C', 'X', 'C']})
        with pytest.raises(ValueError, match='State'):
            _ah_wh_cumulative(
                data,
                method='column',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )


class TestAhWhThroughput:

    def test_invalid_method_raises(self, two_step_ahwh):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_throughput(
                two_step_ahwh,
                method='bogus',
                seconds_alias='Seconds',
                value_alias='Amps',
                valueh_alias='Ah',
            )

    def test_integral_uses_absolute_value(self, two_step_ahwh):
        result = _ah_wh_throughput(
            two_step_ahwh,
            method='integral',
            seconds_alias='Seconds',
            value_alias='Amps',
            valueh_alias='Ah',
        )

        # throughput must be monotonically non-decreasing, regardless of sign
        values = result.to_numpy()
        assert np.all(np.diff(values) >= 0)
        assert values[-1] > 0

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    def test_ah_column_matches_hand_verified_values(self, data):
        result = _ah_wh_throughput(
            data,
            method='column',
            seconds_alias='Seconds',
            value_alias='Amps',
            valueh_alias='Ah',
        )

        expected = data['ThroughputAh'].to_numpy()
        npt.assert_allclose(result.to_numpy(), expected)

    def test_ah_column_negative_value_raises(self):
        data = amp.Dataset({'Ah': [0., -1., 2.]})
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_throughput(
                data,
                method='column',
                seconds_alias='Seconds',
                value_alias='Amps',
                valueh_alias='Ah',
            )
