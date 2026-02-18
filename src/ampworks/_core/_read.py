from __future__ import annotations

from warnings import warn
from typing import Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from ampworks import Dataset


# Define expected headers and their aliases
def format_alias(names: Sequence[str], units: Sequence[str]) -> list[str]:

    aliases = units.copy()  # units only

    for n in names:
        aliases.append(f"{n}")  # name only

        for u in units:
            aliases.append(f"{n}.{u}")  # name.units

    return aliases


t_names = ['t', 'time', 'testtime']
t_units = ['s', 'sec', 'seconds', 'min', 'minutes', 'h', 'hrs', 'hours']

i_names = ['i', 'amperage', 'current']
i_units = ['a', 'amps', 'ma', 'milliamps']

v_names = ['voltage', 'potential', 'ecell']
v_units = ['v', 'volts']

q_names = ['capacity']
q_units = ['ah', 'ahr', 'amphr', 'mah', 'mahr', 'mamphr']

e_names = ['energy']
e_units = ['wh', 'whr', 'watthr']

HEADER_ALIASES = {
    'Seconds': format_alias(t_names, t_units),
    'Amps': format_alias(i_names, i_units),
    'Volts': format_alias(v_names, v_units),

    'Cycle': ['cycle', 'cyc', 'cyclec', 'cyclep', 'cycleindex', 'cyclenumber'],
    'Step': ['step', 'ns', 'stepindex'],
    'State': ['state', 'md'],

    'Ah': format_alias(q_names, q_units),
    'Wh': format_alias(e_names, e_units),

    'DateTime': ['datetime', 'dpttime'],
}

REQUIRED_HEADERS = ['Seconds', 'Amps', 'Volts']


# Remove unnecessary characters from header strings
def strip_chars(string: str) -> str:
    transmap = str.maketrans('(/', '..', ' _-#<>)')
    return string.lower().translate(transmap)


# Matches input headers with aliases of the standard headers
def header_matches(headers: list[str], target_aliases: list[str]) -> bool:
    headers = [strip_chars(h) for h in headers]

    checks = {}
    for k in target_aliases:
        if any(alias in headers for alias in HEADER_ALIASES[k]):
            checks[k] = True
        else:
            checks[k] = False

    return all(checks.values())


# Standardizes the column header names and the data units
def standardize_headers(data: pd.DataFrame) -> Dataset:
    from ampworks import Dataset

    df = Dataset()

    UNIT_FACTORS = {
        'Amps': {
            ('ma', 'mamps', 'milliamps'): 0.001,
        },
        'Ah': {
            ('mah', 'mahr', 'mamphr'): 0.001,
        },
        'Seconds': {
            ('min', 'mins', 'minute', 'minutes'): 60.,
            ('h', 'hr', 'hrs', 'hour', 'hours'): 3600.,
        },
    }

    # Match as-imported headers with standardized headers
    for std_header in HEADER_ALIASES.keys():
        for h1 in data.columns:
            h2 = strip_chars(h1)
            if h2 not in HEADER_ALIASES[std_header]:
                continue

            df[std_header] = data[h1]

            # Standardize units
            if std_header in UNIT_FACTORS.keys():
                for units, factor in UNIT_FACTORS[std_header].items():
                    if any(u in h2 for u in units):
                        df[std_header] = df[std_header].astype(float)*factor
                        break

    # Create 'State' data if not present
    if ('State' not in df.columns) and ('Amps' in df.columns):
        df['Amps'] = df['Amps'].astype(float)

        df['State'] = 'R'
        df.loc[df['Amps'] > 0, 'State'] = 'C'
        df.loc[df['Amps'] < 0, 'State'] = 'D'

    # Guarantee sign 'Amps' sign convention (+ charge, - discharge)
    if 'State' in df.columns:
        df['Amps'] = df['Amps'].astype(float)

        sign = df['State'].map({'R': 0., 'C': +1, 'D': -1}).fillna(1)
        df['Amps'] = sign*df['Amps'].abs()

    # Create 'Ah' and 'Wh' from separate charge and discharge columns
    if any(h not in df.columns for h in ['Ah', 'Wh']):
        Q_headers = ['charge' + h for h in HEADER_ALIASES['Ah']]
        E_headers = ['charge' + h for h in HEADER_ALIASES['Wh']]
        for h1 in data.columns:
            h2 = strip_chars(h1)
            if h2 in Q_headers:
                df['Ah'] = data[h1]
                discharge_Ah = data[h1.replace('Charge', 'Discharge')]
                df.loc[df['State'] == 'D', 'Ah'] = discharge_Ah
            if h2 in E_headers:
                df['Wh'] = data[h1]
                discharge_Wh = data[h1.replace('Charge', 'Discharge')]
                df.loc[df['State'] == 'D', 'Wh'] = discharge_Wh

    # Final data typing, unit normalization, and checks for missing headers
    missing = []
    for std_header in HEADER_ALIASES.keys():

        # Convert types
        if std_header in df.columns:
            if std_header in ['DateTime', 'State']:
                df[std_header] = df[std_header].astype(str)
            elif std_header in ['Cycle', 'Step']:
                df[std_header] = df[std_header].astype(int)
            else:
                df[std_header] = df[std_header].replace('#', '', regex=True)
                df[std_header] = df[std_header].replace(',', '', regex=True)
                df[std_header] = df[std_header].astype(float)
        else:
            missing.append(std_header)

    if missing:
        warn(f"No valid headers found for std_header={missing}.")

    return df


def read_table(filepath: PathLike) -> Dataset:
    """Read tab-delimited file."""
    from ampworks import Dataset

    with open(filepath, encoding='utf-8') as datafile:

        skiprows, found_header = 0, False
        for idx, line in enumerate(datafile):
            if header_matches(line.rstrip('\n').split('\t'), REQUIRED_HEADERS):
                skiprows, found_header = idx, True
                break

        if found_header:
            df = pd.read_csv(filepath, sep='\t', skiprows=skiprows,
                             encoding_errors='ignore')

            return standardize_headers(df)

    warn(f"No valid headers found in {filepath}")
    return Dataset()


def read_excel(
    filepath: PathLike,
    sheet_name: str | int | list[int, str] | None = None,
    stack_sheets: bool = False,
) -> Dataset:
    """Read excel file."""
    from ampworks import Dataset

    workbook = pd.ExcelFile(filepath)
    all_sheets = workbook.sheet_names
    num_sheets = len(all_sheets)

    # warn if 'all' matches a sheet name
    if sheet_name == 'all' and 'all' in all_sheets:
        warn()

    # Set which sheets to iterate through
    if sheet_name is None or sheet_name == 'all':
        iter_sheets = all_sheets
    elif isinstance(sheet_name, (str, int)):
        iter_sheets = [sheet_name]
    elif isinstance(sheet_name, Sequence):
        iter_sheets = list(sheet_name)
    else:
        raise TypeError(
            "'sheet_name' expected a str, int, list[str, int], or None, but"
            f" got {type(sheet_name)}."
        )

    # Raise errors if invalid indices/names
    indices = [v for v in iter_sheets if isinstance(v, int)]
    strings = [v for v in iter_sheets if isinstance(v, str)]
    others = [v for v in iter_sheets if not isinstance(v, (int, str))]

    bad_ind = [i for i in indices if not -num_sheets <= i < num_sheets]
    bad_str = [s for s in strings if s not in all_sheets]
    if bad_ind:
        raise ValueError(f"Invalid sheet indices {bad_ind}, has {num_sheets}")
    if bad_str:
        raise ValueError(f"Invalid worksheet names {bad_str}")
    if others:
        raise TypeError("'sheet_name' must only contain str and/or int types")

    # Iterate through select sheets
    failed = []
    datasets = {}
    for sheet in iter_sheets:
        preview = workbook.parse(sheet, header=None, nrows=20, dtype=str)

        # Find header row
        header_row = None
        for idx, row in preview.iterrows():
            tmp_headers = row.fillna('NaN').astype(str).values
            if header_matches(tmp_headers, REQUIRED_HEADERS):
                header_row = idx
                break

        if header_row is not None:
            df = workbook.parse(sheet, header=header_row)
            datasets[sheet] = standardize_headers(df)
            if sheet_name is None:
                break
        else:
            failed.append(sheet)

    # Prepare outputs
    if sheet_name is None and failed:
        warn(f"Could not find valid headers in requested sheets: {failed}")

    if not datasets:
        warn(f"No valid headers found in requested sheets of {filepath}")
        return Dataset()

    if stack_sheets:
        stack = pd.concat([ds for ds in datasets.values()], ignore_index=True)
        return Dataset(stack)

    if len(datasets) == 1:
        (single,) = datasets.values()
        return single

    return datasets


def read_csv(filepath: PathLike) -> Dataset:
    """Read csv file."""
    from ampworks import Dataset

    with open(filepath, encoding='utf-8') as datafile:

        skiprows, found_header = 0, False
        for idx, line in enumerate(datafile):
            if header_matches(line.rstrip('\n').split(','), REQUIRED_HEADERS):
                skiprows, found_header = idx, True
                break

        if found_header:
            df = pd.read_csv(filepath, sep=',', skiprows=skiprows,
                             encoding_errors='ignore')

            return standardize_headers(df)

    warn(f"No valid headers found in {filepath}")
    return Dataset()
