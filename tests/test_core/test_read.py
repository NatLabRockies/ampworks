import pathlib

from warnings import filterwarnings

import pytest
import pandas as pd
import ampworks as amp


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

    # Aliases with str, list[str], and None
    aliases = amp.HeaderAliases(
        Seconds='elapsed_s',
        Amps=['amps_raw'],
        Volts=None,
    )

    data = reader(file, aliases=aliases, extra_columns={'Meta': 'string'})

    assert data['Meta'].to_list() == ['a', 'b']
    assert pd.api.types.is_string_dtype(data['Meta'])
    assert set(['Seconds', 'Amps', 'Volts', 'Meta']).issubset(data.columns)
