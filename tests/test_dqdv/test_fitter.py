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
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter()

    # returns None for empty input
    ocv, dvdq = fitter._build_splines(None, 'dummy')
    assert ocv is None
    assert dvdq is None

    # soc orientation is opposite for neg vs pos and cell (datasets are dischg)
    assert gr['SOC'].iloc[0] == 0 and gr['SOC'].iloc[-1] == 1
    assert nmc['SOC'].iloc[0] == 0 and nmc['SOC'].iloc[-1] == 1
    assert cell['SOC'].iloc[0] == 0 and cell['SOC'].iloc[-1] == 1

    def flip_and_sort(df):
        df_flipped = df.copy()
        df_flipped['SOC'] = 1 - df_flipped['SOC']
        return df_flipped.sort_values('SOC', ignore_index=True)

    def check_voltage_direction(ocv, key):
        v0, v1 = ocv([0, 1])
        if key == 'neg':
            assert v0 > v1  # voltage should decrease with SOC for neg
        else:
            assert v0 < v1  # voltage should increase with SOC for pos and cell

    # negative electrode
    gr_dis = gr.copy()
    gr_chg = flip_and_sort(gr_dis)

    ocv_d, _ = fitter._build_splines(gr_dis, 'neg')
    ocv_c, _ = fitter._build_splines(gr_chg, 'neg')

    check_voltage_direction(ocv_d, 'neg')
    check_voltage_direction(ocv_c, 'neg')

    assert fitter._initialized['neg']

    # positive electrode
    nmc_dis = nmc.copy()
    nmc_chg = flip_and_sort(nmc_dis)

    ocv_d, _ = fitter._build_splines(nmc_dis, 'pos')
    ocv_c, _ = fitter._build_splines(nmc_chg, 'pos')

    check_voltage_direction(ocv_d, 'pos')
    check_voltage_direction(ocv_c, 'pos')

    assert fitter._initialized['pos']

    # full cell
    cell_dis = cell.copy()
    cell_chg = flip_and_sort(cell_dis)

    ocv_d, _ = fitter._build_splines(cell_dis, 'cell')
    ocv_c, _ = fitter._build_splines(cell_chg, 'cell')

    check_voltage_direction(ocv_d, 'cell')
    check_voltage_direction(ocv_c, 'cell')

    assert fitter._initialized['cell']


def test_check_initialized():
    fitter = amp.dqdv.DqdvFitter()

    # should raise if not initialized
    with pytest.raises(RuntimeError, match="Can't run 'dummy'"):
        fitter._check_initialized('dummy')

    # should not raise if all initialized
    fitter._initialized = {key: True for key in fitter._initialized}
    fitter._check_initialized('dummy')


def test_err_func(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)
    fitter.cost_terms = 'voltage'
    err1 = fitter._err_func([0, 1, 0, 1, 0])

    fitter.cost_terms = ['voltage', 'dqdv']
    err2 = fitter._err_func([0, 1, 0, 1, 0])

    fitter.cost_terms = ['voltage', 'dqdv', 'dvdq']
    err3 = fitter._err_func([0, 1, 0, 1, 0])

    assert err1 < err2 < err3  # more terms should increase error

    fitter.cost_terms = 'all'
    err4 = fitter._err_func([0, 1, 0, 1, 0])

    assert err4 == err3  # 'all' should be same as all terms


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
    fitter = amp.dqdv.DqdvFitter()

    # only works for 'neg', 'pos', 'cell'
    with pytest.raises(ValueError):
        _ = fitter.get_ocv('invalid', [0, 1])

    with pytest.raises(ValueError):
        _ = fitter.get_dqdv('invalid', [0, 1])

    with pytest.raises(ValueError):
        _ = fitter.get_dvdq('invalid', [0, 1])

    # fails if not initialized
    with pytest.raises(RuntimeError):
        _ = fitter.get_ocv('neg', [0, 1])

    with pytest.raises(RuntimeError):
        _ = fitter.get_dqdv('pos', [0, 1])

    with pytest.raises(RuntimeError):
        _ = fitter.get_dvdq('cell', [0, 1])

    # works when initialized
    fitter.neg = datasets['gr_s']
    _ = fitter.get_ocv('neg', [0, 1])
    _ = fitter.get_dqdv('neg', [0, 1])
    _ = fitter.get_dvdq('neg', [0, 1])


def test_err_terms(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)

    # works for params of length 4 or 5
    _ = fitter.err_terms([0, 1, 0, 1])
    _ = fitter.err_terms([0, 1, 0, 1, 0])

    # fails with longer or shorter params
    with pytest.raises(ValueError):
        _ = fitter.err_terms([0, 1, 0])

    with pytest.raises(ValueError):
        _ = fitter.err_terms([0, 1, 0, 1, 0, 1])


def test_grid_search(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)

    # should list iR, but have zero value, all std estimates should be nan
    result1 = fitter.grid_search(Nx=3)

    assert 'iR' in result1.x_map
    assert isinstance(result1, amp.dqdv.DqdvFitResult)

    iR_idx = result1.x_map.index('iR')

    assert result1.x[iR_idx] == 0.
    assert np.all(np.isnan(result1.x_std))

    # higher Nx should have more evaluations
    result2 = fitter.grid_search(Nx=5)
    assert result1.nfev < result2.nfev

    # best answer should have lower error than full range
    result3 = fitter.grid_search(Nx=11)
    assert result3.fun < fitter._err_func([0, 1, 0, 1, 0])

    # all x values should still be in [0, 1]
    assert np.all((0 <= result3.x) & (result3.x <= 1))


def test_constrainted_fit(datasets):
    from scipy.optimize import OptimizeResult

    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)

    # bounds type error - must be float or list[float]
    with pytest.raises(TypeError):
        _ = fitter.constrained_fit([0, 1, 0, 1], bounds='invalid')

    # bounds length should be four, if not float
    with pytest.raises(ValueError):
        _ = fitter.constrained_fit([0, 1, 0, 1], bounds=[0.01, 0.01])

    # bounds are respected
    x0 = np.array([0, 1, 0, 1])
    result = fitter.constrained_fit(x0, bounds=0.01)
    assert np.all(np.abs(np.round(result.x[:4] - x0, 2)) <= 0.01)

    # iR is fit when 'voltage' in cost_terms
    iR_idx = result.x_map.index('iR')

    assert 'voltage' in fitter.cost_terms
    assert result.x[iR_idx] != 0.
    assert result.x_std[iR_idx] != 0.

    # iR is not fit when 'voltage' not in cost_terms
    fitter.cost_terms = ['dqdv', 'dvdq']
    result = fitter.constrained_fit(x0, bounds=0.01)

    assert result.x[iR_idx] == 0.
    assert result.x_std[iR_idx] == 0.

    # std estimates are in result
    assert not np.any(np.isnan(result.x_std))

    # all x values should still be in [0, 1]
    assert np.all((0 <= result.x) & (result.x <= 1))

    # best answer should have lower error than starting point, and return_full
    x0 = [0.0, 0.8, 0.0, 0.9, 0.0]  # close to optimal
    fit_result, opt_result = fitter.constrained_fit(x0, return_full=True)
    assert isinstance(fit_result, amp.dqdv.DqdvFitResult)
    assert isinstance(opt_result, OptimizeResult)
    assert fit_result.fun < fitter._err_func(x0)


def test_plot(datasets):
    gr, nmc, cell = datasets['gr_s'], datasets['nmc_s'], datasets['cell1_s']

    fitter = amp.dqdv.DqdvFitter(gr, nmc, cell)

    # no output when not requested
    assert fitter.plot([0, 1, 0, 1], return_axs=False) is None

    # works with length 4 or 5 inputs
    axs = fitter.plot([0, 1, 0, 1], return_axs=True)
    assert isinstance(axs, list)
    assert len(axs) == 4

    axs = fitter.plot([0, 1, 0, 1, 0], return_axs=True)
    assert isinstance(axs, list)
    assert len(axs) == 4

    # should have correct number of lines on each plot
    assert len(axs[0].lines) == 5  # voltage: data, model, pos, ref lines
    assert len(axs[1].lines) == 1  # voltage: neg plotted on twin axis
    assert len(axs[2].lines) == 2  # dqdv: data + model
    assert len(axs[3].lines) == 2  # dvdq: data

    def get_labels(ax):
        labels = [line.get_label() for line in ax.lines]
        return set([label for label in labels if not label.startswith('_')])

    assert get_labels(axs[0]) == {'Data', 'Model', 'Pos'}
    assert get_labels(axs[1]) == {'Neg'}
    assert get_labels(axs[2]) == {'Data', 'Model'}
    assert get_labels(axs[3]) == {'Data', 'Model'}

    # compatible with fit result object
    result = amp.dqdv.DqdvFitResult(x=[0, 1, 0, 1, 0])
    axs = fitter.plot(result.x, return_axs=True)
    assert isinstance(axs, list)
    assert len(axs) == 4
