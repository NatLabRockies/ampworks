import pytest
import numpy.testing as npt

import ampworks as amp

from ampworks.columns import add_power, add_c_rate


@pytest.fixture
def rates_data():
    """Amps/Volts pairs with hand-computed Watts and CRate expectations."""
    return amp.Dataset({
        'Amps': [1., -2., 0., 3.],
        'Volts': [4.0, 3.5, 4.2, 3.0],
        'ExpectedWatts': [4.0, -7.0, 0.0, 9.0],
        'ExpectedCRate': [1., -2., 0., 3.],  # amps_1c = 1.0
    })


class TestAddPower:

    def test_missing_columns_raises(self, rates_data):
        ds = rates_data.drop(columns=['Volts'])
        with pytest.raises(ValueError):
            add_power(ds)

    def test_values(self, rates_data):
        result = add_power(rates_data)
        npt.assert_allclose(result['Watts'], rates_data['ExpectedWatts'])

    def test_col_name(self, rates_data):
        result = add_power(rates_data, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'Watts' not in result.columns

    def test_aliases(self, rates_data):
        ds = rates_data.rename(columns={'Amps': 'Current', 'Volts': 'Voltage'})
        result = add_power(ds, amps_alias='Current', volts_alias='Voltage')
        npt.assert_allclose(result['Watts'], rates_data['ExpectedWatts'])

    def test_returns_copy(self, rates_data):
        result = add_power(rates_data)
        assert result is not rates_data
        assert 'Watts' not in rates_data.columns


class TestAddCRate:

    def test_missing_columns_raises(self, rates_data):
        ds = rates_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            add_c_rate(ds, amps_1c=1.0)

    def test_values(self, rates_data):
        result = add_c_rate(rates_data, amps_1c=1.0)
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_amps_1c_uses_magnitude(self, rates_data):
        # sign of amps_1c shouldn't matter, only its magnitude
        result = add_c_rate(rates_data, amps_1c=-1.0)
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_col_name(self, rates_data):
        result = add_c_rate(rates_data, amps_1c=1.0, col_name='Custom')
        assert 'Custom' in result.columns
        assert 'CRate' not in result.columns

    def test_amps_alias(self, rates_data):
        ds = rates_data.rename(columns={'Amps': 'Current'})
        result = add_c_rate(ds, amps_1c=1.0, amps_alias='Current')
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_returns_copy(self, rates_data):
        result = add_c_rate(rates_data, amps_1c=1.0)
        assert result is not rates_data
        assert 'CRate' not in rates_data.columns
