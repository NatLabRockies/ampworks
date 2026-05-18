import pytest
import numpy as np
import pandas as pd
import ampworks as amp


@pytest.fixture(scope='module')
def datasets():
    datasets = {}
    datasets['charge'] = amp.datasets.load_datasets('ici/ici_charge')
    datasets['discharge'] = amp.datasets.load_datasets('ici/ici_discharge')
    return datasets


def test_extract_params_missing_columns():

    ds = amp.Dataset({'Seconds': [], 'Volts': []})  # missing 'Amps'
    with pytest.raises(ValueError):
        _ = amp.ici.extract_params(ds, 1.8e-6)


def test_extract_params_charge_discharge(datasets):
    ds = datasets['discharge'].copy()

    ds.loc[0, 'Amps'] = +1  # inject opposite sign
    with pytest.raises(ValueError):
        _ = amp.ici.extract_params(ds, 1.8e-6)


def test_extract_params_basic(datasets):
    ds = datasets['discharge'].copy()

    # test with discharge data, with return_stats=True
    params, stats = amp.ici.extract_params(ds, 1.8e-6, return_stats=True)

    assert isinstance(params, pd.DataFrame)
    assert {'SOC', 'Ds', 'Eeq'}.issubset(params.columns)

    assert isinstance(stats, pd.DataFrame)
    assert stats.shape[0] == params.shape[0]

    assert params.shape[0] >= 100
    assert params.notna().values.all()
    assert params['SOC'].is_monotonic_increasing

    assert np.all((params['SOC'] >= 0) & (params['SOC'] <= 1))
    assert np.all((params['Ds'] < 4e-15) & (params['Ds'] > 1e-16))
    assert np.all((params['Eeq'] >= 3.0) & (params['Eeq'] <= 4.1))

    # test with charge data - overwrite "ds" fixture
    ds = datasets['charge'].copy()

    params = amp.ici.extract_params(ds, 1.8e-6)

    assert isinstance(params, pd.DataFrame)
    assert {'SOC', 'Ds', 'Eeq'}.issubset(params.columns)

    assert params.shape[0] >= 100
    assert params.notna().values.all()
    assert params['SOC'].is_monotonic_increasing

    assert np.all((params['SOC'] >= 0) & (params['SOC'] <= 1))
    assert np.all((params['Ds'] < 4e-15) & (params['Ds'] > 1e-16))
    assert np.all((params['Eeq'] >= 3.0) & (params['Eeq'] <= 4.1))


def test_extract_params_truncate_last_step(datasets):
    from ampworks._auxiliary import _infer_state

    ds = datasets['discharge'].copy()

    _infer_state(ds)

    rest = (ds['State'] != 'R') & (ds['State'].shift(fill_value='R') == 'R')
    ds['Rest'] = rest.cumsum()

    # get first 5 discharge/rest steps, then remove last rest
    subset = ds[ds['Rest'] <= 5]
    last_rest = subset[(subset['Rest'] == 5) & (subset['State'] == 'R')]
    subset = subset.drop(last_rest.index)

    params = amp.ici.extract_params(subset, 1.8e-6)

    assert params.shape[0] == 4  # one less than 5 b/c incomplete step cut off
    assert params.notna().values.all()  # shouldn't have NaN for complete steps


def test_extract_params_rest_not_in_twindow(datasets):
    from ampworks._auxiliary import _infer_state

    ds = datasets['discharge'].copy()

    # if tmin and tmax are set such that the number of points is less than two
    # then the linear regression cannot be performed and NaN is returned
    _infer_state(ds)

    rest = (ds['State'] != 'R') & (ds['State'].shift(fill_value='R') == 'R')
    ds['Rest'] = rest.cumsum()

    # get first 5 pulse/rest steps, then make last rest points all < tmin=1
    subset = ds[ds['Rest'] <= 5]

    last = subset[(subset['Rest'] == 5) & (subset['State'] == 'R')]
    tmin_limit = last[last['Seconds'] - last['Seconds'].iloc[0] > 1]

    subset = subset.drop(tmin_limit.index)

    params = amp.ici.extract_params(subset, 1.8e-6)

    assert params.shape[0] == 5

    # expect NaNs in first row (lowest SOC b/c discharge)
    assert params['SOC'].is_monotonic_increasing
    assert np.isnan(params['Eeq'].iloc[0])
    assert np.isnan(params['Ds'].iloc[0])

    # remaining rows should not have any NaN
    assert params[1:].notna().values.all()
