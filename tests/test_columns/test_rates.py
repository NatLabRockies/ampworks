import pytest
import numpy.testing as npt

import ampworks as amp
import ampworks.columns as col


@pytest.fixture
def rates_data():
    """Amps/Volts pairs with hand-computed Watts and CRate expectations."""
    return amp.Dataset({
        'Amps': [1., -2., 0., 3.],
        'Volts': [4.0, 3.5, 4.2, 3.0],
        'ExpectedWatts': [4.0, -7.0, 0.0, 9.0],
        'ExpectedCRate': [0.5, -1., 0., 1.5],  # nominal_ah = 2.0
    })


class TestAddPower:

    def test_missing_columns_raises(self, rates_data):
        ds = rates_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            col.add_power(ds)

        ds = rates_data.drop(columns=['Volts'])
        with pytest.raises(ValueError):
            col.add_power(ds)

    def test_values(self, rates_data):
        result = col.add_power(rates_data)
        npt.assert_allclose(result['Watts'], rates_data['ExpectedWatts'])

    def test_output_col(self, rates_data):
        result = col.add_power(rates_data, output_col='Custom')
        assert 'Custom' in result.columns
        assert 'Watts' not in result.columns

    def test_custom_cols(self, rates_data):
        ds = rates_data.rename(columns={'Amps': 'Current', 'Volts': 'Voltage'})
        result = col.add_power(ds, amps_col='Current', volts_col='Voltage')
        npt.assert_allclose(result['Watts'], rates_data['ExpectedWatts'])

    def test_returns_copy(self, rates_data):
        result = col.add_power(rates_data)
        assert result is not rates_data
        assert 'Watts' not in rates_data.columns


class TestAddCRate:

    def test_missing_columns_raises(self, rates_data):
        ds = rates_data.drop(columns=['Amps'])
        with pytest.raises(ValueError):
            col.add_c_rate(ds, nominal_ah=2.0)

    def test_values(self, rates_data):
        result = col.add_c_rate(rates_data, nominal_ah=2.0)
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_nominal_ah_uses_magnitude(self, rates_data):
        # sign of nominal_ah shouldn't matter, only its magnitude
        result = col.add_c_rate(rates_data, nominal_ah=-2.0)
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_zero_nominal_ah_raises(self, rates_data):
        with pytest.raises(ValueError, match='nonzero'):
            col.add_c_rate(rates_data, nominal_ah=0.0)

    def test_output_col(self, rates_data):
        result = col.add_c_rate(rates_data, nominal_ah=2.0, output_col='Custom')
        assert 'Custom' in result.columns
        assert 'CRate' not in result.columns

    def test_amps_col(self, rates_data):
        ds = rates_data.rename(columns={'Amps': 'Current'})
        result = col.add_c_rate(ds, nominal_ah=2.0, amps_col='Current')
        npt.assert_allclose(result['CRate'], rates_data['ExpectedCRate'])

    def test_returns_copy(self, rates_data):
        result = col.add_c_rate(rates_data, nominal_ah=2.0)
        assert result is not rates_data
        assert 'CRate' not in rates_data.columns
