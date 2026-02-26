import pytest
import numpy as np
import ampworks as amp


@pytest.fixture
def datasets():
    dqdv_datasets = amp.datasets.list_datasets('dqdv')

    data = {}
    for name in dqdv_datasets:
        key = name.removeprefix('dqdv/').removesuffix('.csv')
        key = key.replace('_smooth', '_s')
        key = key.replace('_rough', '_r')

        data[key] = amp.datasets.load_datasets(name)
        data[key]['SOC'] = data[key]['Ah'] / data[key]['Ah'].max()

    return data


def test_check_dataframe(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter()

    # should reset initialization flags
    fitter._initialized['neg'] = True
    _ = fitter._check_dataframe(gr, 'neg')
    assert not fitter._initialized['neg']

    # should return None for empty input
    assert fitter._check_dataframe(None, 'neg') is None

    # only works with None or dataframes
    with pytest.raises(TypeError):
        _ = fitter._check_dataframe('invalid', 'neg')

    # missing columns (all require {'SOC', 'Volts'}, cell also needs 'Ah')
    dummy = gr.drop(columns='SOC')
    with pytest.raises(ValueError):
        _ = fitter._check_dataframe(dummy, 'neg')

    dummy = nmc.drop(columns='Volts')
    with pytest.raises(ValueError):
        _ = fitter._check_dataframe(dummy, 'pos')

    dummy = cell.drop(columns='Ah')
    with pytest.raises(ValueError):
        _ = fitter._check_dataframe(dummy, 'cell')


def test_build_splines(datasets):
    # TODO: make sure soc orientation is correct
    # TODO: make sure initialized flags are set to True
    ...


def test_check_initialized(datasets):
    # TODO: make sure functions raise errors if data is needed but not set
    ...


def test_err_func(datasets):
    # TODO: make sure values are different when cost_terms is changed
    ...


def test_data_setters(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    def check_is_set(fitter, key):
        assert getattr(fitter, key) is not None
        assert callable(getattr(fitter, f"_ocv_{key[0]}"))
        assert callable(getattr(fitter, f"_dvdq_{key[0]}"))

        if key == 'cell':
            assert isinstance(fitter._soc, np.ndarray)
            assert fitter._soc.min() == 0.
            assert fitter._soc.max() == 1.

            assert isinstance(fitter._volt_data, np.ndarray)
            assert isinstance(fitter._dqdv_data, np.ndarray)
            assert isinstance(fitter._dvdq_data, np.ndarray)

    # set at initialization
    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)
    check_is_set(fitter, 'neg')
    check_is_set(fitter, 'pos')
    check_is_set(fitter, 'cell')

    # set after initialization
    fitter = amp.dqdv.DqdvFitter()

    fitter.neg = gr
    check_is_set(fitter, 'neg')

    fitter.pos = nmc
    check_is_set(fitter, 'pos')

    fitter.cell = cell
    check_is_set(fitter, 'cell')

    # cell requires an 'Ah' column
    dummy = cell.drop(columns='Ah')
    with pytest.raises(ValueError):
        _ = amp.dqdv.DqdvFitter(gr, nmc, dummy)

    fitter = amp.dqdv.DqdvFitter()
    with pytest.raises(ValueError):
        fitter.cell = dummy


def test_cost_terms_setter():
    fitter = amp.dqdv.DqdvFitter()

    # single value becomes a list
    fitter.cost_terms = 'all'
    assert isinstance(fitter.cost_terms, list)
    assert set(fitter.cost_terms) == {'voltage', 'dqdv', 'dvdq'}

    fitter.cost_terms = 'voltage'
    assert fitter.cost_terms == ['voltage']

    # must be a non-empty sequence
    with pytest.raises(TypeError):
        fitter.cost_terms = {'voltage': True}  # not a sequence

    with pytest.raises(ValueError):
        fitter.cost_terms = []  # cannot be empty

    # only valid with subsets of {'voltage', 'dqdv', 'dvdq'}
    with pytest.raises(ValueError):
        fitter.cost_terms = ['invalid']


def test_get_funcs(datasets):
    # TODO: consider using the get funcs internally to grid_search, etc.
    # TODO: make sure get_ocv, get_dqdv, and get_dvdq work correctly
    ...


def test_err_terms(datasets):
    # TODO: make sure errors are percents even though err_func is fractional
    ...


def test_grid_search(datasets):
    # TODO: make sure more evaluations when Nx is changed
    # TODO: evaluate that the returned values are better than full range
    # TODO: make sure iR is listed, but is always zero
    # TODO: verify that std estimates are present, but are nan
    ...


def test_constrainted_fit(datasets):
    # TODO: check that bounds are respected
    # TODO: ensure iR term is only fit when 'voltage' is in cost_terms
    # TODO: verify that std estimates are in result
    ...


def test_plot(datasets):
    # TODO: works with length 4 or 5 inputs
    # TODO: compatible with fit results from grid_search and constrained_fit
    ...
