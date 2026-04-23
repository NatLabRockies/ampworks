import pathlib

from warnings import filterwarnings

import pytest
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

    assert data['Notes'].dtype == 'string'
    assert data['Temperature'].dtype == 'float64'
    assert data['Notes'].to_list() == ['start', 'run']


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

    aliases = amp.HeaderAliases(
        Seconds='elapsed_s',
        Amps='amps_raw',
        Volts='volts_raw',
    )

    data = reader(file, aliases=aliases, extra_columns={'Meta': None})

    assert set(['Seconds', 'Amps', 'Volts', 'Meta']).issubset(data.columns)
    assert data['Meta'].to_list() == ['a', 'b']
