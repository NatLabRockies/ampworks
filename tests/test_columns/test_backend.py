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
def repeated_steps():
    """Step 1 and Step 2 alternate twice, all within a single cycle."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 1, 1],
        'Cycle': [1, 1, 1, 1, 1, 1],
    })


@pytest.fixture
def two_cycle_repeats():
    """The repeated_steps pattern repeated a second time in a new cycle."""
    return amp.Dataset({
        'Step': [1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1],
        'Cycle': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
    })


@pytest.fixture
def two_step_current():
    """Two steps, each with its own constant current, uniform 1 s spacing."""
    return amp.Dataset({
        'Step': [1, 1, 1, 2, 2, 2],
        'Seconds': [0., 1., 2., 3., 4., 5.],
        'Amps': [1., 1., 1., -2., -2., -2.],
    })


REST_STEP = amp.Dataset({  # includes a rest step between active segments
    'Ah': [0, 1, 3, 7, 0, 0, 0, 0, 2, 3],
    'State1': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'C', 'C', 'C'],
    'State2': ['C', 'C', 'C', 'C', 'R', 'R', 'R', 'D', 'D', 'D'],
    'State3': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'C', 'C', 'C'],
    'State4': ['D', 'D', 'D', 'D', 'R', 'R', 'R', 'D', 'D', 'D'],
    'CumulativeAh1': [0, 1, 3, 7, 7, 7, 7, 7, 9, 10],
    'CumulativeAh2': [0, 1, 3, 7, 7, 7, 7, 7, 5, 4],
    'CumulativeAh3': [0, -1, -3, -7, -7, -7, -7, -7, -5, -4],
    'CumulativeAh4': [0, -1, -3, -7, -7, -7, -7, -7, -9, -10],
    'ThroughputAh': [0, 1, 3, 7, 7, 7, 7, 7, 9, 10],
})

NO_REST = amp.Dataset({  # goes directly from one active step to another
    'Ah': [0, 1, 3, 7, 0, 2, 3],
    'State1': ['C', 'C', 'C', 'C', 'C', 'C', 'C'],
    'State2': ['C', 'C', 'C', 'C', 'D', 'D', 'D'],
    'State3': ['D', 'D', 'D', 'D', 'C', 'C', 'C'],
    'State4': ['D', 'D', 'D', 'D', 'D', 'D', 'D'],
    'CumulativeAh1': [0, 1, 3, 7, 7, 9, 10],
    'CumulativeAh2': [0, 1, 3, 7, 7, 5, 4],
    'CumulativeAh3': [0, -1, -3, -7, -7, -5, -4],
    'CumulativeAh4': [0, -1, -3, -7, -7, -9, -10],
    'ThroughputAh': [0, 1, 3, 7, 7, 9, 10],
})


class TestInstanceNums:

    def test_dense_rank_within_group(self, repeated_steps):
        result = _instance_nums(
            repeated_steps,
            which='Step',
            cycle_alias=None,
            cycle_resets=False,
        )
        npt.assert_array_equal(result.to_numpy(), [1, 1, 1, 1, 2, 2])

    def test_fast_returns_raw_changeover_counts(self, repeated_steps):
        result = _instance_nums(
            repeated_steps,
            which='Step',
            cycle_alias=None,
            cycle_resets=False,
            fast=True,
        )
        npt.assert_array_equal(result.to_numpy(), [1, 1, 2, 2, 3, 3])

    def test_cycle_resets_requires_cycle_alias(self, two_cycle_repeats):
        with pytest.raises(ValueError, match='cycle_alias'):
            _instance_nums(
                two_cycle_repeats,
                which='Step',
                cycle_alias=None,
                cycle_resets=True,
            )

    def test_cycle_resets_false_is_global(self, two_cycle_repeats):
        result = _instance_nums(
            two_cycle_repeats,
            which='Step',
            cycle_alias=None,
            cycle_resets=False,
        )
        expected = [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]
        npt.assert_array_equal(result.to_numpy(), expected)

    def test_cycle_resets_true_restarts_per_cycle(self, two_cycle_repeats):
        result = _instance_nums(
            two_cycle_repeats,
            which='Step',
            cycle_alias='Cycle',
            cycle_resets=True,
        )
        expected = [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2]
        npt.assert_array_equal(result.to_numpy(), expected)

    def test_fast_with_cycle_resets_warns_and_ignores_reset(
        self, two_cycle_repeats,
    ):
        with pytest.warns(UserWarning, match='fast=True'):
            result = _instance_nums(
                two_cycle_repeats,
                which='Step',
                cycle_alias=None,
                cycle_resets=True,
                fast=True,
            )

        # cycle boundary alone isn't a changeover in 'Step', so it merges with
        # the following block since raw only reacts to 'which' value changes
        expected = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5]
        npt.assert_array_equal(result.to_numpy(), expected)


class TestAhWh:

    def test_missing_columns_raises(self, two_step_current):
        with pytest.raises(ValueError):
            _ah_wh(
                two_step_current,
                which='Step',
                col_name='Ah',
                seconds_alias='Missing',
                value_alias='Amps',
            )

    def test_resets_and_integrates_per_group(self, two_step_current):
        result = _ah_wh(
            two_step_current,
            which='Step',
            col_name='Ah',
            seconds_alias='Seconds',
            value_alias='Amps',
        )

        expected = np.array([0., 1., 2., 0., -2., -4.]) / 3600.
        npt.assert_allclose(result['Ah'].to_numpy(), expected)

    def test_returns_copy(self, two_step_current):
        result = _ah_wh(
            two_step_current,
            which='Step',
            col_name='Ah',
            seconds_alias='Seconds',
            value_alias='Amps',
        )
        assert result is not two_step_current
        assert 'Ah' not in two_step_current.columns


class TestAhWhCumulative:

    def test_invalid_method_raises(self, two_step_current):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_cumulative(
                two_step_current,
                method_options={'integral', 'ah_column'},
                method='bogus',
                col_name='Result',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )

    def test_integral_method_does_not_require_state_column(
        self,
        two_step_current,
    ):
        # state_alias/valueh_alias are unused (and absent) for 'integral'
        result = _ah_wh_cumulative(
            two_step_current,
            method_options={'integral', 'ah_column'},
            method='integral',
            col_name='Result',
            seconds_alias='Seconds',
            value_alias='Amps',
            state_alias='State',
            valueh_alias='Ah',
        )

        hours = np.array([0., 1., 2., 3., 4., 5.]) / 3600.
        amps = np.array([1., 1., 1., -2., -2., -2.])
        expected = np.concatenate([[0.], np.cumsum(
            (amps[1:] + amps[:-1]) / 2 * np.diff(hours)
        )])
        npt.assert_allclose(result['Result'].to_numpy(), expected)

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    @pytest.mark.parametrize('n', [1, 2, 3, 4])
    def test_ah_column_matches_hand_verified_values(self, data, n):
        result = _ah_wh_cumulative(
            data,
            method_options={'integral', 'ah_column'},
            method='ah_column',
            col_name='Result',
            seconds_alias='Seconds',
            value_alias='Amps',
            state_alias=f'State{n}',
            valueh_alias='Ah',
        )

        expected = data[f'CumulativeAh{n}'].to_numpy()
        npt.assert_allclose(result['Result'].to_numpy(), expected)

    def test_ah_column_negative_value_raises(self):
        data = amp.Dataset({
            'Ah': [0., -1., 2.],
            'State': ['C', 'C', 'C'],
        })
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_cumulative(
                data,
                method_options={'integral', 'ah_column'},
                method='ah_column',
                col_name='Result',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )

    def test_ah_column_invalid_state_raises(self):
        data = amp.Dataset({
            'Ah': [0., 1., 2.],
            'State': ['C', 'X', 'C'],
        })
        with pytest.raises(ValueError, match='State'):
            _ah_wh_cumulative(
                data,
                method_options={'integral', 'ah_column'},
                method='ah_column',
                col_name='Result',
                seconds_alias='Seconds',
                value_alias='Amps',
                state_alias='State',
                valueh_alias='Ah',
            )


class TestAhWhThroughput:

    def test_invalid_method_raises(self, two_step_current):
        with pytest.raises(ValueError, match='method'):
            _ah_wh_throughput(
                two_step_current,
                method_options={'integral', 'ah_column'},
                method='bogus',
                col_name='Result',
                seconds_alias='Seconds',
                value_alias='Amps',
                valueh_alias='Ah',
            )

    def test_integral_uses_absolute_value(self, two_step_current):
        result = _ah_wh_throughput(
            two_step_current,
            method_options={'integral', 'ah_column'},
            method='integral',
            col_name='Result',
            seconds_alias='Seconds',
            value_alias='Amps',
            valueh_alias='Ah',
        )

        # throughput must be monotonically non-decreasing, regardless of sign
        values = result['Result'].to_numpy()
        assert np.all(np.diff(values) >= 0)
        assert values[-1] > 0

    @pytest.mark.parametrize('data', [REST_STEP, NO_REST])
    def test_ah_column_matches_hand_verified_values(self, data):
        result = _ah_wh_throughput(
            data,
            method_options={'integral', 'ah_column'},
            method='ah_column',
            col_name='Result',
            seconds_alias='Seconds',
            value_alias='Amps',
            valueh_alias='Ah',
        )

        expected = data['ThroughputAh'].to_numpy()
        npt.assert_allclose(result['Result'].to_numpy(), expected)

    def test_ah_column_negative_value_raises(self):
        data = amp.Dataset({'Ah': [0., -1., 2.]})
        with pytest.raises(ValueError, match='non-negative'):
            _ah_wh_throughput(
                data,
                method_options={'integral', 'ah_column'},
                method='ah_column',
                col_name='Result',
                seconds_alias='Seconds',
                value_alias='Amps',
                valueh_alias='Ah',
            )
