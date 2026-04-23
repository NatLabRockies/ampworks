from __future__ import annotations

import csv
import textwrap

from warnings import warn
from typing import Sequence, Generator, TYPE_CHECKING

import pandas as pd
import polars as pl

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


# Remove unnecessary characters from header strings
def strip_chars(string: str | list[str] | None) -> str | list[str] | None:
    if string is None:
        return None
    elif isinstance(string, list):
        return [strip_chars(s) for s in string]

    transmap = str.maketrans('(/,', '...', ' _-#<>)')
    return string.lower().translate(transmap)


t_names = ['t', 'time', 'testtime', 'totaltime']
t_units = ['s', 'sec', 'seconds', 'min', 'minutes', 'h', 'hrs', 'hours']

i_names = ['i', 'amperage', 'current']
i_units = ['a', 'amps', 'ma', 'milliamps']

v_names = ['voltage', 'potential', 'ecell']
v_units = ['v', 'volts']

q_names = ['capacity', 'amphours']
q_units = ['ah', 'ahr', 'amphr', 'mah', 'mahr', 'mamphr']

e_names = ['energy', 'watthours']
e_units = ['wh', 'whr', 'watthr']

HEADER_ALIASES = {
    'Seconds': format_alias(t_names, t_units),
    'Amps': format_alias(i_names, i_units),
    'Volts': format_alias(v_names, v_units),

    'Cycle': ['cycle', 'cyc', 'cycleindex', 'cyclenumber', 'cyclec', 'cyclep'],
    'Step': ['step', 'ns', 'stepindex'],
    'State': ['state', 'md', 'mode'],

    'Ah': format_alias(q_names, q_units),
    'Wh': format_alias(e_names, e_units),

    'DateTime': ['datetime', 'dpttime', 'realtime'],
}

REQUIRED_HEADERS = ['Seconds', 'Amps', 'Volts']


# Allow users to define their own custom header aliases
class HeaderAliases:

    __slots__ = ('Seconds', 'Amps', 'Volts', 'Cycle', 'Step', 'State', 'Ah',
                 'Wh', 'DateTime')

    def __init__(
        self,
        Seconds: str | list[str] | None = None,
        Amps: str | list[str] | None = None,
        Volts: str | list[str] | None = None,
        Cycle: str | list[str] | None = None,
        Step: str | list[str] | None = None,
        State: str | list[str] | None = None,
        Ah: str | list[str] | None = None,
        Wh: str | list[str] | None = None,
        DateTime: str | list[str] | None = None,
    ) -> None:
        from ampworks._checks import _check_type, _check_inner_type

        params = {
            'Seconds': Seconds,
            'Amps': Amps,
            'Volts': Volts,
            'Cycle': Cycle,
            'Step': Step,
            'State': State,
            'Ah': Ah,
            'Wh': Wh,
            'DateTime': DateTime,
        }

        # convert inputs to list[str] or use defaults if None
        def make_list_or_default(key, value):
            if value is None:
                return HEADER_ALIASES[key]
            if isinstance(value, str):
                return strip_chars([value])
            return strip_chars(value)

        # loop over fields and add to class instance
        for name, value in params.items():

            _check_type(name, value, (str, list, None))
            value = make_list_or_default(name, value)
            _check_inner_type(name, value, str)

            setattr(self, name, value)

    def __getitem__(self, key):
        if key in self.__slots__:
            return getattr(self, key)
        raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __repr__(self) -> str:  # pragma: no cover

        data = {k: v for k, v in self.items()}

        summary = "\n".join([f"{k}={v!r}," for k, v in data.items()])
        summary = textwrap.indent(summary, " " * 4)

        return f"{self.__class__.__name__}(\n{summary}\n)"

    def keys(self) -> list[str]:
        return list(self.__slots__)

    def items(self) -> Generator:
        for slot in self.__slots__:
            yield (slot, getattr(self, slot))


# Matches input headers with aliases of the standard headers
def header_matches(
    headers: list[str], targets: list[str], aliases: HeaderAliases,
) -> bool:
    headers = strip_chars(headers)

    checks = {}
    for k in targets:
        if any(alias in headers for alias in aliases[k]):
            checks[k] = True
        else:
            checks[k] = False

    return all(checks.values())


# Standardizes the column header names and the data units
def standardize_headers(
    data: pd.DataFrame,
    aliases: HeaderAliases | None = None,
    extra_columns: dict[str, type | None] | None = None,
) -> Dataset:
    from ampworks import Dataset

    print(data.columns)
    print(data.dtypes)

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
    for std_header in aliases.keys():
        for h1 in data.columns:
            h2 = strip_chars(h1)
            if h2 not in aliases[std_header]:
                continue

            if std_header not in df.columns:
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
        rename_bitrode = {'REST': 'R', 'DCHG': 'D', 'CHRG': 'C'}
        df['State'] = df['State'].replace(rename_bitrode)

        df['Amps'] = df['Amps'].astype(float)
        df['State'] = df['State'].astype(str)

        sign = df['State'].map({'R': 0., 'C': +1, 'D': -1}).fillna(1)
        df['Amps'] = sign*df['Amps'].abs()

    # Create 'Ah' and 'Wh' from separate charge and discharge columns
    if any(h not in df.columns for h in ['Ah', 'Wh']):
        Q_headers = ['charge' + h for h in aliases['Ah']]
        E_headers = ['charge' + h for h in aliases['Wh']]
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
    for std_header in aliases.keys():

        # Convert types
        if std_header in df.columns:
            if std_header in ['State', 'DateTime']:
                df[std_header] = df[std_header].astype('string')
            elif std_header in ['Cycle', 'Step']:
                df[std_header] = df[std_header].astype('Int64')
            else:
                df[std_header] = df[std_header].replace('#', '', regex=True)
                df[std_header] = df[std_header].replace(',', '', regex=True)

                df[std_header] = pd.to_numeric(df[std_header], errors='coerce')
        else:
            missing.append(std_header)

    if missing:
        warn(f"No valid aliases found for {missing}.")

    # Keep user-requested non-standardized columns from source data.
    if extra_columns is not None:
        missing_extra = []
        skipped_extra = []

        for col_name, col_type in extra_columns.items():
            if col_name not in data.columns:
                missing_extra.append(col_name)
                continue

            if col_name in df.columns:
                skipped_extra.append(col_name)
                continue

            df[col_name] = data[col_name]
            if col_type is not None:
                df[col_name] = df[col_name].astype(col_type)

        if missing_extra:
            warn(f"'extra_columns' not found: {missing_extra=}. Only found"
                 f" {set(data.columns)}.")

        if skipped_extra:
            warn(f"Skipped some conflicting 'extra_columns': {skipped_extra=}."
                 f" Existing are {set(df.columns)}.")

    return df


def read_table(
    filepath: PathLike,
    aliases: HeaderAliases | None = None,
    extra_columns: dict[str, type | None] | None = None,
) -> Dataset:
    """
    Read a tab-delimited file.

    Custom reading function for tab-delimited files. Scans the file to identify
    expected headers (see Notes for specifics). This routine is not specific to
    any particular cycler. Instead, it uses internal or user-defined aliases to
    find and standardize the headers, columns, and data types.

    Parameters
    ----------
    filepath : PathLike
        Path to the file. Must include the file extension.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Additional columns to include in the standardized dataset. Include both
        the exact source column names and their corresponding data types in a
        dictionary. Use value None to keep pandas-inferred dtype. The `type` is
        also compatible with pandas dtypes, e.g., 'string', 'Int64', etc.

    Returns
    -------
    dataset : Dataset
        The read data.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers.

    See Also
    --------
    Dataset
    HeaderAliases

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    """
    from ampworks import Dataset
    from ampworks._checks import _check_type

    if aliases is None:
        aliases = HeaderAliases()

    _check_type('aliases', aliases, HeaderAliases)

    options = {'separator': '\t', 'skip_rows': 0, 'ignore_errors': True}
    with open(filepath, encoding='latin1') as datafile:
        reader = csv.reader(datafile, delimiter='\t')

        found_header = False
        for idx, line in enumerate(reader):
            if header_matches(line, REQUIRED_HEADERS, aliases):
                options['skip_rows'] = idx
                found_header = True
                break

        if found_header:
            df = pl.read_csv(filepath, **options).to_pandas()
            return standardize_headers(df, aliases, extra_columns)

    return Dataset()


def read_excel(
    filepath: PathLike,
    sheet_name: str | int | list[int, str] | None = None,
    stack_sheets: bool = False,
    aliases: HeaderAliases | None = None,
    extra_columns: dict[str, type | None] | None = None,
) -> Dataset:
    """
    Read an Excel file.

    Custom reading function for Excel files. Scans all (or some) of the sheets
    to identify expected headers (see Notes for specifics). This routine is not
    specific to any particular cycler. Instead, it uses internal or user-defined
    aliases to find and standardize the headers, columns, and data types.

    Parameters
    ----------
    filepath : PathLike
        Path to the file. Must include the file extension.
    sheet_name : str or int or list[str, int] or None
        Name or index of the sheet(s) to read. For integers, use natural indices
        from 1 to the number of sheets. None (default) will scan for the first
        sheet with valid headers. Use `'all'` to read all sheets.
    stack_sheets : bool
        If True, concatenate all read sheets into a single dataset.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Additional columns to include in the standardized dataset. Include both
        the exact source column names and their corresponding data types in a
        dictionary. Use value None to keep pandas-inferred dtype. The `type` is
        also compatible with pandas dtypes, e.g., 'string', 'Int64', etc.

    Returns
    -------
    dataset : Dataset or dict[Dataset]
        The read data. If many sheets are read without stacking, a dictionary of
        Datasets is returned. The keys correspond to the provided `sheet_name`
        values, so may be a mix of strings and integers.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers.

    See Also
    --------
    Dataset
    HeaderAliases

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    """
    from ampworks import Dataset
    from ampworks._checks import _check_type, _check_inner_type

    workbook = pd.ExcelFile(filepath)
    all_sheets = workbook.sheet_names
    num_sheets = len(all_sheets)

    # warn if 'all' matches a sheet name
    if sheet_name == 'all' and 'all' in all_sheets:
        warn("sheet_name='all' is interpreted as ALL sheets, but a sheet named"
             " 'all' exists. To read only that sheet, pass ['all'] explicitly.")

    # Set which sheets to iterate through
    _check_type('sheet_name', sheet_name, (str, int, Sequence, None))

    if sheet_name is None or sheet_name == 'all':
        iter_sheets = all_sheets
    elif isinstance(sheet_name, (str, int)):
        iter_sheets = [sheet_name]
    elif isinstance(sheet_name, Sequence):
        iter_sheets = list(sheet_name)

    # Raise errors if invalid indices/names
    _check_inner_type('sheet_name', iter_sheets, (str, int))

    strings = [v for v in iter_sheets if isinstance(v, str)]
    indices = [v for v in iter_sheets if isinstance(v, int)]

    bad_str = [s for s in strings if s not in all_sheets]
    bad_ind = [i for i in indices if not 1 <= i <= num_sheets]

    if bad_str:
        raise ValueError(f"Invalid worksheet names {bad_str}.")
    if bad_ind:
        raise ValueError(f"Invalid sheet indices {bad_ind}, must be between 1"
                         f" and {num_sheets}.")

    # Set up aliases or use defaults
    if aliases is None:
        aliases = HeaderAliases()

    _check_type('aliases', aliases, HeaderAliases)

    # Iterate through select sheets
    failed = []
    datasets = {}
    for sheet in iter_sheets:
        preview = workbook.parse(sheet, header=None, nrows=20, dtype=str)

        # Find header row
        header_row = None
        for idx, row in preview.iterrows():
            tmp_headers = row.fillna('NaN').astype(str).to_list()
            if header_matches(tmp_headers, REQUIRED_HEADERS, aliases):
                header_row = idx
                break

        if header_row is not None:
            sheet_int = sheet if isinstance(sheet, int) else None
            sheet_str = sheet if isinstance(sheet, str) else None
            read_options = {'header_row': header_row} if header_row > 0 else {}

            df = pl.read_excel(
                filepath,
                sheet_id=sheet_int,
                sheet_name=sheet_str,
                read_options=read_options,
            )

            datasets[sheet] = standardize_headers(
                df.to_pandas(), aliases, extra_columns,
            )

            if sheet_name is None:
                break
        else:
            failed.append(sheet)

    # Prepare outputs
    if sheet_name is None and failed:
        warn(f"No valid aliases found in requested sheets: {failed}.")

    if not datasets:
        warn(f"No valid aliases found in requested sheets of {filepath}.")
        return Dataset()

    if stack_sheets:
        stack = pd.concat([ds for ds in datasets.values()], ignore_index=True)
        return Dataset(stack)

    if len(datasets) == 1:
        (single,) = datasets.values()
        return single

    return datasets


def read_csv(
    filepath: PathLike,
    aliases: HeaderAliases | None = None,
    extra_columns: dict[str, type | None] | None = None,
) -> Dataset:
    """
    Read a csv file.

    Custom reading function for comma-separated values (CSV) files. Scans the
    file to identify expected headers (see Notes for specifics). This routine is
    not specific to any particular cycler. Instead, it uses default internal or
    user-defined aliases to find and standardize the headers, columns, and data
    types.

    Parameters
    ----------
    filepath : PathLike
        Path to the file. Must include the file extension.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Additional columns to include in the standardized dataset. Include both
        the exact source column names and their corresponding data types in a
        dictionary. Use value None to keep pandas-inferred dtype. The `type` is
        also compatible with pandas dtypes, e.g., 'string', 'Int64', etc.

    Returns
    -------
    dataset : Dataset
        The read data.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers.

    See Also
    --------
    Dataset
    HeaderAliases

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    """
    from ampworks import Dataset
    from ampworks._checks import _check_type

    if aliases is None:
        aliases = HeaderAliases()

    _check_type('aliases', aliases, HeaderAliases)

    options = {'separator': ',', 'skip_rows': 0, 'ignore_errors': True}
    with open(filepath, encoding='latin1') as datafile:
        reader = csv.reader(datafile, delimiter=',')

        found_header = False
        for idx, line in enumerate(reader):
            if header_matches(line, REQUIRED_HEADERS, aliases):
                options['skip_rows'] = idx
                found_header = True
                break

        if found_header:
            df = pl.read_csv(filepath, **options).to_pandas()
            return standardize_headers(df, aliases, extra_columns)

        else:
            warn(f"No valid aliases found for {REQUIRED_HEADERS} in {filepath}."
                 " Returning empty dataset.")

    return Dataset()
