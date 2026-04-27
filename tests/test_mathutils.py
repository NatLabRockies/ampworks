import pytest
import ampworks as amp

import numpy as np
import numpy.testing as npt


def make_dataset(x_vals, y_vals, x_col, y_col):
    return amp.Dataset({x_col: np.array(x_vals), y_col: np.array(y_vals)})


def test_aggregate_over_x_raises():

    # No overlap in x column
    data1 = make_dataset([0.0, 1.0], [0.0, 1.0], x_col='Volts', y_col='Ah')
    data2 = make_dataset([2.0, 3.0], [2.0, 3.0], x_col='Volts', y_col='Ah')

    with pytest.raises(ValueError, match='No overlapping range'):
        amp.mathutils.aggregate_over_x([data1, data2], x='Volts', y='Ah')

    # Missing one of the required x/y columns
    data1 = make_dataset([0.0, 1.0], [0.0, 1.0], x_col='Volts', y_col='Ah')
    data2 = amp.Dataset({'Volts': np.array([0.0, 1.0])})

    with pytest.raises(ValueError, match='requires'):
        amp.mathutils.aggregate_over_x([data1, data2], x='Volts', y='Ah')

    # Invalid type, not a sequence
    data1 = make_dataset([0.0, 1.0], [0.0, 1.0], x_col='Volts', y_col='Ah')

    with pytest.raises(TypeError):
        amp.mathutils.aggregate_over_x(data1, x='Volts', y='Ah')

    # Needs at least one dataset
    with pytest.raises(ValueError, match='at least one dataset'):
        amp.mathutils.aggregate_over_x([], x='Volts', y='Ah')

    # Grid size must be at least 2
    data1 = make_dataset([0.0, 1.0], [0.0, 1.0], x_col='Volts', y_col='Ah')

    with pytest.raises(ValueError, match='at least 2'):
        amp.mathutils.aggregate_over_x([data1], x='Volts', y='Ah', n=1)


@pytest.mark.parametrize('input_type', [list, tuple])
def test_aggregate_over_x_returns(input_type):
    data1 = make_dataset([0.0, 1.0], [0.0, 2.0], x_col='Volts', y_col='Ah')
    data2 = make_dataset([0.0, 1.0], [0.0, 4.0], x_col='Volts', y_col='Ah')

    expected = {'Volts', 'Ah_mean', 'Ah_std', 'Ah_min', 'Ah_max'}

    # Works with list/tuple for inputs sequence
    datasets = input_type([data1, data2])
    out = amp.mathutils.aggregate_over_x(datasets, x='Volts', y='Ah', n=3)

    # Output type is correct and has expected columns
    assert isinstance(out, amp.Dataset)
    assert set(out.columns) == expected

    # Size of output is correct
    assert len(out) == 3


def test_aggregate_over_x_expected_stats():

    # Perfect overlap
    data1 = make_dataset([0.0, 1.0], [0.0, 2.0], x_col='Volts', y_col='Ah')
    data2 = make_dataset([0.0, 1.0], [0.0, 4.0], x_col='Volts', y_col='Ah')

    out = amp.mathutils.aggregate_over_x([data1, data2], x='Volts', y='Ah')

    assert len(out) == 100
    npt.assert_allclose(out['Volts'].to_numpy(), np.linspace(0.0, 1.0, 100))

    assert out['Ah_mean'].iloc[0] == 0.0
    assert out['Ah_std'].iloc[0] == np.std([0.0, 0.0], ddof=1)
    assert out['Ah_min'].iloc[0] == 0.0
    assert out['Ah_max'].iloc[0] == 0.0

    assert out['Ah_mean'].iloc[-1] == 3.0
    assert out['Ah_std'].iloc[-1] == np.std([2.0, 4.0], ddof=1)
    assert out['Ah_min'].iloc[-1] == 2.0
    assert out['Ah_max'].iloc[-1] == 4.0

    # Overlap only spans part of each x range, use more than two datasets
    data1 = make_dataset([0.0, 1.0], [0.0, 1.0], x_col='Volts', y_col='Ah')
    data2 = make_dataset([0.5, 1.5], [1.0, 3.0], x_col='Volts', y_col='Ah')

    datasets = [data1, data1, data2, data2]
    out = amp.mathutils.aggregate_over_x(datasets, x='Volts', y='Ah')

    assert len(out) == 100
    npt.assert_allclose(out['Volts'].to_numpy(), np.linspace(0.5, 1.0, 100))

    assert out['Ah_mean'].iloc[0] == np.mean([0.5, 0.5, 1.0, 1.0])
    assert out['Ah_std'].iloc[0] == np.std([0.5, 0.5, 1.0, 1.0], ddof=1)
    assert out['Ah_min'].iloc[0] == np.min([0.5, 0.5, 1.0, 1.0])
    assert out['Ah_max'].iloc[0] == np.max([0.5, 0.5, 1.0, 1.0])

    assert out['Ah_mean'].iloc[-1] == np.mean([1.0, 1.0, 2.0, 2.0])
    assert out['Ah_std'].iloc[-1] == np.std([1.0, 1.0, 2.0, 2.0], ddof=1)
    assert out['Ah_min'].iloc[-1] == np.min([1.0, 1.0, 2.0, 2.0])
    assert out['Ah_max'].iloc[-1] == np.max([1.0, 1.0, 2.0, 2.0])

    # Works with a single dataset, std should be zero
    out = amp.mathutils.aggregate_over_x([data1], x='Volts', y='Ah')

    assert len(out) == 100
    npt.assert_allclose(out['Volts'].to_numpy(), np.linspace(0.0, 1.0, 100))

    npt.assert_allclose(out['Ah_mean'].to_numpy(), np.linspace(0.0, 1.0, 100))
    npt.assert_allclose(out['Ah_std'].to_numpy(), np.zeros(100))
    npt.assert_allclose(out['Ah_min'].to_numpy(), np.linspace(0.0, 1.0, 100))
    npt.assert_allclose(out['Ah_max'].to_numpy(), np.linspace(0.0, 1.0, 100))


def test_combinations():

    names = ['a', 'b']
    values = [np.array([0., 1.]), np.array([3., 4.])]

    combinations_nonames = amp.mathutils.combinations(values)
    combinations_names = amp.mathutils.combinations(values, names)

    no_names = []
    with_names = []
    for a in values[0]:
        new_nonames, new_names = {}, {}
        new_nonames[0], new_names['a'] = a, a
        for b in values[1]:
            new_nonames[1], new_names['b'] = b, b

            no_names.append(new_nonames.copy())
            with_names.append(new_names.copy())

    assert combinations_nonames == no_names
    assert combinations_names == with_names
