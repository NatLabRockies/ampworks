import pytest
import ampworks as amp

import numpy as np
import numpy.testing as npt


@pytest.fixture
def sample_data():
    """A Dataset with columns A and Seconds, 10 rows, and nonzero start."""
    return amp.Dataset({
        'A': [3., 1., 4., 1., 5., 9., 2., 6., 5., 3.],
        'Seconds': [100., 110., 120., 130., 140., 150., 160., 170., 180., 190.],
    })


@pytest.fixture
def monotonic_data():
    """A Dataset with strictly increasing I and decreasing D, respectively."""
    return amp.Dataset({
        'I': [1., 2., 3., 4., 5.],
        'D': [5., 4., 3., 2., 1.],
        'Seconds': [0., 1., 2., 3., 4.],
    })


@pytest.fixture
def noisy_data():
    """A Dataset where A is mostly increasing but has dips."""
    return amp.Dataset({
        'A': [1., 3., 2., 5., 5., 4., 6., 8., 7., 9., 10., 10.],
        'Seconds': [0., 1., 3., 2., 5., 4., 6., 8., 7., 9., 10., 11.],
    })


# downsample method validation and functionality
class TestDownsampleValidation:

    def test_empty_dataset(self):
        empty = amp.Dataset({'A': [], 'Seconds': []})
        with pytest.raises(ValueError):
            empty.downsample(n=5)

    def test_all_none_raises(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample()

    def test_two_params_raises(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(n=5, frac=0.5)

    def test_three_params_raises(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(n=5, frac=0.5, resolution=('A', 0.1))

    def test_n_zero(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(n=0)

    def test_n_negative(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(n=-1)

    def test_frac_zero(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(frac=0.0)

    def test_frac_negative(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(frac=-0.1)

    def test_frac_above_one(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(frac=1.5)

    def test_resolution_not_length_two(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(resolution=('A',))

    def test_resolution_missing_column(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.downsample(resolution=('missing_col', 0.1))


class TestDownsampleFunctional:

    def test_n_returns_correct_row_count(self, sample_data):
        result = sample_data.downsample(n=5)
        assert len(result) == 5

    def test_n_larger_than_dataset_clips(self, sample_data):
        result = sample_data.downsample(n=len(sample_data) + 10)
        assert len(result) == len(sample_data)

    def test_n_first_row_kept(self, sample_data):
        result = sample_data.downsample(n=3)
        assert result.iloc[0].equals(sample_data.iloc[0])

    def test_frac_returns_correct_row_count(self, sample_data):
        result = sample_data.downsample(frac=0.5)
        expected = int(len(sample_data) * 0.5)
        assert len(result) == expected

    def test_frac_one_returns_all_rows(self, sample_data):
        result = sample_data.downsample(frac=1.0)
        assert len(result) == len(sample_data)

    def test_small_frac_keeps_at_least_one_row(self, sample_data):
        result = sample_data.downsample(frac=0.01)
        assert len(result) >= 1

    def test_resolution_adjacent_gap_satisfied(self, sample_data):
        atol = 3.0

        assert np.any(sample_data['A'].diff().abs() < atol)  # before

        result = sample_data.downsample(resolution=('A', atol))
        values = result['A'].to_numpy()

        assert len(result) <= len(sample_data)  # at least some rows dropped
        assert np.all(np.abs(np.diff(values)) >= atol)  # after

    def test_resolution_first_row_kept(self, sample_data):
        result = sample_data.downsample(resolution=('A', 3.0))
        assert result.iloc[0].equals(sample_data.iloc[0])

    def test_keeping_last_row(self, sample_data):
        result = sample_data.downsample(n=1, keep_last=True)
        assert result.iloc[-1].equals(sample_data.iloc[-1])

        result = sample_data.downsample(n=1, keep_last=False)
        assert not result.iloc[-1].equals(sample_data.iloc[-1])

    def test_inplace_true_returns_none(self, sample_data):
        result = sample_data.downsample(n=5, inplace=True)
        assert result is None

    def test_inplace_true_modifies_self(self, sample_data):
        original_len = len(sample_data)
        sample_data.downsample(n=5, inplace=True)
        assert len(sample_data) < original_len
        assert len(sample_data) == 5

    def test_inplace_false_leaves_original_unchanged(self, sample_data):
        original_len = len(sample_data)
        _ = sample_data.downsample(n=5, inplace=False)
        assert len(sample_data) == original_len

    def test_ignore_index_true(self, sample_data):
        result = sample_data.downsample(n=3, ignore_index=True)
        assert list(result.index) == list(range(len(result)))

    def test_ignore_index_false(self, sample_data):
        result = sample_data.downsample(n=5, ignore_index=False)
        assert np.any(result.index.diff() > 1)

    def test_return_type_is_dataset(self, sample_data):
        result = sample_data.downsample(n=5)
        assert isinstance(result, amp.Dataset)


# enforce_monotonic method validation and functionality
class TestEnforceMonotonicValidation:

    def test_empty_dataset(self):
        empty = amp.Dataset({'A': []})
        with pytest.raises(ValueError):
            empty.enforce_monotonic(column='A')

    def test_missing_column(self, sample_data):
        with pytest.raises(ValueError):
            sample_data.enforce_monotonic(column='missing_col')


class TestEnforceMonotonicFunctional:

    def test_increasing_nonstrict(self, noisy_data):
        first, last = noisy_data['A'].iloc[[0, -1]].to_numpy()

        if first > last:
            noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        assert np.any(noisy_data['A'].diff() < 0)  # sanity check

        result = noisy_data.enforce_monotonic(
            column='A', increasing=True, strict=False,
        )

        # at least some rows dropped, remaining non-strictly increasing
        assert len(result) <= len(noisy_data)
        assert np.all(np.diff(result['A'].to_numpy()) >= 0)

    def test_increasing_strict(self, noisy_data):
        first, last = noisy_data['A'].iloc[[0, -1]].to_numpy()

        if first > last:
            noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        assert np.any(noisy_data['A'].diff() < 0)  # sanity checks
        assert np.any(noisy_data['A'].diff() == 0)

        result = noisy_data.enforce_monotonic(
            column='A', increasing=True, strict=True,
        )

        # at least some rows dropped, remaining strictly increasing
        assert len(result) <= len(noisy_data)
        assert np.all(np.diff(result['A'].to_numpy()) > 0)

    def test_decreasing_nonstrict(self, noisy_data):
        first, last = noisy_data['A'].iloc[[0, -1]].to_numpy()

        if first < last:
            noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        assert np.any(noisy_data['A'].diff() > 0)  # sanity check

        result = noisy_data.enforce_monotonic(
            column='A', increasing=False, strict=False,
        )

        # at least some rows dropped, remaining non-strictly decreasing
        assert len(result) <= len(noisy_data)
        assert np.all(np.diff(result['A'].to_numpy()) <= 0)

    def test_decreasing_strict(self, noisy_data):
        first, last = noisy_data['A'].iloc[[0, -1]].to_numpy()

        if first < last:
            noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        noisy_data['A'] = np.flip(noisy_data['A'].to_numpy())

        assert np.any(noisy_data['A'].diff() > 0)  # sanity checks
        assert np.any(noisy_data['A'].diff() == 0)

        result = noisy_data.enforce_monotonic(
            column='A', increasing=False, strict=True,
        )

        # at least some rows dropped, remaining strictly decreasing
        assert len(result) <= len(noisy_data)
        assert np.all(np.diff(result['A'].to_numpy()) < 0)

    def test_already_monotonic(self, monotonic_data):
        result = monotonic_data.enforce_monotonic(column='I', increasing=True)
        assert len(result) == len(monotonic_data)

        result = monotonic_data.enforce_monotonic(column='D', increasing=False)
        assert len(result) == len(monotonic_data)

    def test_all_equal_keep_at_least_first_row(self):
        data = amp.Dataset({'A': [3.0, 3.0, 3.0, 3.0]})
        result = data.enforce_monotonic(
            column='A', increasing=True, strict=False,
        )
        assert len(result) == 4

        result = data.enforce_monotonic(
            column='A', increasing=True, strict=True,
        )
        assert len(result) == 1

    def test_inplace_true_returns_none(self, noisy_data):
        result = noisy_data.enforce_monotonic(column='A', inplace=True)
        assert result is None

    def test_inplace_true_modifies_self(self, noisy_data):
        original_len = len(noisy_data)
        noisy_data.enforce_monotonic(column='A', inplace=True)
        assert len(noisy_data) <= original_len

    def test_inplace_false_leaves_original_unchanged(self, noisy_data):
        original_len = len(noisy_data)
        _ = noisy_data.enforce_monotonic(column='A', inplace=False)
        assert len(noisy_data) == original_len

    def test_ignore_index_true(self, noisy_data):
        result = noisy_data.enforce_monotonic(column='A', ignore_index=True)
        assert list(result.index) == list(range(len(result)))

    def test_ignore_index_false(self, noisy_data):
        result = noisy_data.enforce_monotonic(column='A', ignore_index=False)
        assert np.any(result.index.diff() > 1)

    def test_return_type_is_dataset(self, noisy_data):
        result = noisy_data.enforce_monotonic(column='A')
        assert isinstance(result, amp.Dataset)


# zero_time method validation and functionality
class TestZeroTimeValidation:

    def test_missing_seconds_column(self):
        data = amp.Dataset({'A': [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError):
            data.zero_time()


class TestZeroTimeFunctional:

    def test_first_row_zero(self, sample_data):
        assert sample_data['Seconds'].iloc[0] != 0.0  # sanity check
        result = sample_data.zero_time()
        assert result['Seconds'].iloc[0] == 0.0

    def test_remaining_rows_shifts(self, sample_data):
        seconds = sample_data['Seconds'].to_numpy()
        assert seconds[0] != 0.0  # sanity check

        result = sample_data.zero_time()
        expected = seconds - seconds[0]

        npt.assert_almost_equal(result['Seconds'].to_numpy(), expected)

    def test_negative_start_time_handled(self):
        data = amp.Dataset({'Seconds': [-50., -30., -10., 10.]})
        result = data.zero_time()
        npt.assert_allclose(result['Seconds'].to_numpy(), [0., 20., 40., 60.])

    def test_other_columns_unchanged(self, sample_data):
        result = sample_data.zero_time()
        npt.assert_allclose(result['A'].to_numpy(), sample_data['A'].to_numpy())

    def test_inplace_true_returns_none(self, sample_data):
        result = sample_data.zero_time(inplace=True)
        assert result is None

    def test_inplace_true_modifies_self(self, sample_data):
        assert sample_data['Seconds'].iloc[0] != 0.0  # sanity check
        sample_data.zero_time(inplace=True)
        assert sample_data['Seconds'].iloc[0] == 0.0

    def test_inplace_false_leaves_original_unchanged(self, sample_data):
        original_first = sample_data['Seconds'].iloc[0]
        assert original_first != 0.0  # sanity check
        _ = sample_data.zero_time(inplace=False)
        assert sample_data['Seconds'].iloc[0] == original_first

    def test_return_type_is_dataset(self, sample_data):
        result = sample_data.zero_time()
        assert isinstance(result, amp.Dataset)


# zero_below method functionality
class TestZeroBelowFunctional:

    def test_values_below_threshold_zeroed(self):
        data = amp.Dataset({'A': [-0.5, -2.0, 0.5, 2.0]})
        result = data.zero_below(column='A', threshold=1.0)
        npt.assert_allclose(result['A'].to_numpy(), [0.0, -2.0, 0.0, 2.0])

    def test_value_equal_to_threshold_not_zeroed(self):
        data = amp.Dataset({'A': [1.0, 2.0, 3.0]})
        result = data.zero_below(column='A', threshold=2.0)
        assert result['A'].iloc[1] == 2.0  # equal to threshold, kept

    def test_values_above_threshold_unchanged(self):
        data = amp.Dataset({'A': [0.5, 5.0, 10.0]})
        result = data.zero_below(column='A', threshold=1.0)
        npt.assert_allclose(result['A'].to_numpy(), [0.0, 5.0, 10.0])

    def test_negative_threshold_same_as_positive(self):
        # abs(threshold) is used, so -1.0 and 1.0 produce identical results
        data = amp.Dataset({'A': [0.5, 1.5, 2.5]})
        pos = data.zero_below(column='A', threshold=1.0)
        neg = data.zero_below(column='A', threshold=-1.0)
        npt.assert_allclose(pos['A'].to_numpy(), neg['A'].to_numpy())

    def test_other_columns_unchanged(self):
        data = amp.Dataset({'A': [0.5, 2.0], 'B': [10.0, 20.0]})
        result = data.zero_below(column='A', threshold=1.0)
        npt.assert_allclose(result['B'].to_numpy(), [10.0, 20.0])

    def test_inplace_true_returns_none(self, sample_data):
        result = sample_data.zero_below(column='A', threshold=1.0, inplace=True)
        assert result is None

    def test_inplace_true_modifies_self(self):
        data = amp.Dataset({'A': [0.5, 1.5, 2.5]})
        data.zero_below(column='A', threshold=1.0, inplace=True)
        assert data['A'].iloc[0] == 0.0
        assert data['A'].iloc[1] == 1.5

    def test_inplace_false_leaves_original_unchanged(self):
        data = amp.Dataset({'A': [0.5, 1.5, 2.5]})
        _ = data.zero_below(column='A', threshold=1.0, inplace=False)
        assert data['A'].iloc[0] == 0.5  # original untouched

    def test_return_type_is_dataset(self, sample_data):
        result = sample_data.zero_below(column='A', threshold=1.0)
        assert isinstance(result, amp.Dataset)
