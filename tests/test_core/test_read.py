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
    file = path / 'dummy_data' / ('sample' + extension)

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
    file = path / 'dummy_data' / ('aliases' + extension)

    all_readers = {
        '.csv': amp.read_csv,
        '.txt': amp.read_table,
        '.xls': amp.read_excel,
        '.xlsx': amp.read_excel,
    }

    reader = all_readers[extension]

    # aliases without conversion or None (use defaults)
    # extend_defaults=False, so Seconds and Amps override defaults
    aliases1 = amp.HeaderAliases(
        Seconds={'elapsed_s': None},
        Amps={'amps_raw': None},
        Volts=None,
        Cycle=['CycleNumber', 'CycleNum'],
        extend_defaults=False,
    )
    data1 = reader(file, aliases=aliases1, extra_columns={'Meta': 'string'})

    assert data1['Meta'].to_list() == ['a', 'b']
    assert pd.api.types.is_string_dtype(data1['Meta'])
    assert set(['Seconds', 'Amps', 'Volts', 'Meta']).issubset(data1.columns)

    # aliases with conversion functions and extend_defaults
    aliases2 = amp.HeaderAliases(
        Seconds={'elapsed_s': 0.5},
        Amps={'amps_raw': lambda x: 2.0 * x},
        Cycle=['CycleNumber', 'CycleNum'],
        extend_defaults=True,
    )
    data2 = reader(file, aliases=aliases2)

    npt.assert_allclose(data2['Seconds'], 0.5*data1['Seconds'])
    npt.assert_allclose(data2['Amps'], 2.0*data1['Amps'])

    # aliases with conversion functions and subset extend_defaults, including
    # spurious 'Volts' which is not in the override aliases anyway...
    aliases3 = amp.HeaderAliases(
        Seconds={'elapsed_s': 0.5},
        Amps={'amps_raw': lambda x: 2.0 * x},
        extend_defaults=['Seconds', 'Amps', 'Volts'],
    )
    data3 = reader(file, aliases=aliases3)

    # data2 and data3 must match because extend_defaults doesn't matter here
    # since the test files use the aliased headers instead of defaults
    assert data2.equals(data3)
