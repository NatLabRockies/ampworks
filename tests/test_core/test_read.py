import pathlib

from warnings import filterwarnings

import pytest
import pandas as pd
import ampworks as amp
import numpy.testing as npt


@pytest.mark.parametrize('extension', ['.csv', '.txt', '.xls', '.xlsx'])
def test_read_extra_columns(extension):
    filterwarnings('ignore', message='.*No valid aliases.*')

    path = pathlib.Path(__file__).parent
    file = path / 'data' / 'dummy' / ('sample' + extension)

    all_readers = {
        '.csv': amp.read_csv,
        '.txt': amp.read_table,
        '.xls': amp.read_excel,
        '.xlsx': amp.read_excel,
    }

    reader = all_readers[extension]

    # extra_columns must be exact match, warn if not found
    with pytest.warns(UserWarning, match="'extra_columns' not found"):
        _ = reader(file, extra_columns={'temperature': None})

    with pytest.warns(UserWarning, match="'extra_columns' not found"):
        _ = reader(file, extra_columns={'missing': None})

    # correct use with no warnings
    data = reader(file, extra_columns={'Temperature': float, 'Notes': None})

    assert {'Temperature', 'Notes'}.issubset(data.columns)

    assert data['Notes'].to_list() == ['start', 'run']
    assert pd.api.types.is_string_dtype(data['Notes'])
    assert pd.api.types.is_float_dtype(data['Temperature'])


@pytest.mark.parametrize('extension', ['.csv', '.txt', '.xls', '.xlsx'])
def test_read_custom_aliases(extension):
    filterwarnings('ignore', message='.*No valid aliases.*')

    path = pathlib.Path(__file__).parent
    file = path / 'data' / 'dummy' / ('aliases' + extension)

    all_readers = {
        '.csv': amp.read_csv,
        '.txt': amp.read_table,
        '.xls': amp.read_excel,
        '.xlsx': amp.read_excel,
    }

    reader = all_readers[extension]

    # aliases without conversion or None (use defaults)
    # extend_defaults=False, so seconds and amps override defaults
    aliases1 = amp.HeaderAliases(
        seconds={'elapsed_s': None},
        amps={'amps_raw': None},
        volts=None,
        cycle=['CycleNumber', 'CycleNum'],
        extend_defaults=False,
    )
    data1 = reader(file, aliases=aliases1, extra_columns={'Meta': 'string'})

    assert data1['Meta'].to_list() == ['a', 'b']
    assert pd.api.types.is_string_dtype(data1['Meta'])
    assert set(['Seconds', 'Amps', 'Volts', 'Meta']).issubset(data1.columns)

    # aliases with conversion functions and extend_defaults
    aliases2 = amp.HeaderAliases(
        seconds={'elapsed_s': 0.5},
        amps={'amps_raw': lambda x: 2.0 * x},
        cycle=['CycleNumber', 'CycleNum'],
        extend_defaults=True,
    )
    data2 = reader(file, aliases=aliases2)

    npt.assert_allclose(data2['Seconds'], 0.5 * data1['Seconds'])
    npt.assert_allclose(data2['Amps'], 2.0 * data1['Amps'])

    # aliases with conversion functions and subset extend_defaults, including
    # spurious 'volts' which is not in the override aliases anyway...
    aliases3 = amp.HeaderAliases(
        seconds={'elapsed_s': 0.5},
        amps={'amps_raw': lambda x: 2.0 * x},
        extend_defaults=['seconds', 'amps', 'volts'],
    )
    data3 = reader(file, aliases=aliases3)

    # data2 and data3 must match because extend_defaults doesn't matter here
    # since the test files use the aliased headers instead of defaults
    assert data2.equals(data3)


@pytest.mark.parametrize('reader, sep', [
    (amp.read_csv, ','),
    (amp.read_table, '\t'),
])
def test_read_non_utf8_and_late_dtype_change(tmp_path, reader, sep):
    # latin-1 bytes (degree sign) in a kept column, and a current column that
    # is integer-like past polars' default 100-row schema inference window
    header = sep.join(['Time (s)', 'Current (A)', 'Voltage (V)', 'Notes'])
    deg_c = '25\xb0C'
    late = [sep.join([str(i), '1.5', '3.6', deg_c]) for i in range(150, 155)]
    rows = [sep.join([str(i), '0', '3.5', 'rest']) for i in range(150)] + late

    file = tmp_path / ('latin1' + ('.csv' if sep == ',' else '.txt'))
    file.write_bytes('\n'.join([header, *rows, '']).encode('latin1'))

    filterwarnings('ignore', message='.*No valid aliases.*')
    data = reader(file, extra_columns={'Notes': None})

    assert len(data) == 155
    assert not data['Amps'].isna().any()
    npt.assert_allclose(data['Amps'].iloc[-5:], 1.5)
    assert data['Notes'].iloc[-1].startswith('25')
