from __future__ import annotations

import csv

from warnings import warn
from typing import TYPE_CHECKING, Sequence

import pandas as pd
import polars as pl

if TYPE_CHECKING:  # pragma: no cover
    from os import PathLike
    from ampworks import Dataset, HeaderAliases


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
        Path to the file, including extension.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Extra source columns to preserve using exact source names as keys. The
        values define cast type. Use None to keep inferred dtype. Both Python
        types and pandas dtypes are accepted, e.g., `'string'`, `'Int64'`, etc.

    Returns
    -------
    dataset : Dataset
        Standardized battery dataset.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers. Also, if no valid headers are found and an
        empty dataset is returned.

    See Also
    --------
    HeaderAliases : Customize the column/header mapping for standardization.

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    Examples
    --------
    The following example shows how to read in data from a `.txt` file using a
    few of the available options.

    .. code-block:: python

        import ampworks as amp

        # read in the file using all default options
        data = amp.read_table('data.txt')

        # specify custom aliases for a couple column headers
        aliases = amp.HeaderAliases(Seconds='Time_s', Amps='Current_A')
        data = amp.read_table('data.txt', aliases=aliases)

        # include extra columns for temperature and notes
        extra_cols = {'Temperature': float, 'Notes': None}
        data = amp.read_table('data.txt', extra_columns=extra_cols)

    """
    from ampworks._checks import _check_type
    from ampworks import Dataset, HeaderAliases
    from ampworks._core._headers import (
        standardize_headers, header_matches, REQUIRED_HEADERS,
    )

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

        warn(f"No valid aliases found for {REQUIRED_HEADERS} in {filepath}."
             " Returning empty dataset.")

    return Dataset()


def read_excel(
    filepath: PathLike,
    sheet_name: str | int | Sequence[str | int] | None = None,
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
        Path to the file, including extension.
    sheet_name : str or int or Sequence[str or int] or None, optional
        Name or index of the sheet(s) to read. For integers, use natural indices
        from 1 to the number of sheets. None (default) will scan for the first
        sheet with valid headers. Use `'all'` to read all sheets.
    stack_sheets : bool, optional
        If True, concatenate all parsed sheets into one dataset.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Extra source columns to preserve using exact source names as keys. The
        values define cast type. Use None to keep inferred dtype. Both Python
        types and pandas dtypes are accepted, e.g., `'string'`, `'Int64'`, etc.

    Returns
    -------
    dataset : Dataset or dict[str or int, Dataset]
        Standardized dataset output. A dictionary is returned if multiple sheets
        are read and `stack_sheets` is False.

    Raises
    ------
    ValueError
        If the parameter (or any of the elements in) `sheet_name` are invalid
        names or indices.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers. Also, if no valid headers are found and an
        empty dataset is returned.

    See Also
    --------
    HeaderAliases : Customize the column/header mapping for standardization.

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    Examples
    --------
    The following example shows how to read in data from an Excel file using a
    few of the available options. Note that the examples demonstrate different
    extensions that are both types of Excel files.

    .. code-block:: python

        import ampworks as amp

        # read in the file using all default options
        data = amp.read_excel('data.xls')

        # specify custom aliases for a couple column headers
        aliases = amp.HeaderAliases(Seconds='Time_s', Amps='Current_A')
        data = amp.read_excel('data.xls', aliases=aliases)

        # include extra columns for temperature and notes
        extra_cols = {'Temperature': float, 'Notes': None}
        data = amp.read_excel('data.xls', extra_columns=extra_cols)

        # specify the second sheet and a sheet named 'last'
        data = amp.read_excel('data.xlsx', sheet_name=[2, 'last'])

        # read in all sheets and concatenate the results
        data = amp.read_excel('data.xlsx', sheet_name='all', stack_sheets=True)

    """
    from ampworks import Dataset, HeaderAliases
    from ampworks._checks import _check_type, _check_inner_type
    from ampworks._core._headers import (
        standardize_headers, header_matches, REQUIRED_HEADERS,
    )

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

    strings = [value for value in iter_sheets if isinstance(value, str)]
    indices = [value for value in iter_sheets if isinstance(value, int)]

    bad_str = [value for value in strings if value not in all_sheets]
    bad_ind = [value for value in indices if not 1 <= value <= num_sheets]

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
        Path to the file, including extension.
    aliases : HeaderAliases or None, optional
        Column alias mapping for the header standardization. If None (default),
        a set of internal default aliases is used.
    extra_columns : dict[str, type or None] or None, optional
        Additional columns to include in the standardized dataset. Include both
        the exact source column names and their corresponding data types in a
        dictionary. Use value None to keep pandas-inferred dtype. The `type` is
        also compatible with pandas dtypes, e.g., `'string'`, `'Int64'`, etc.

    Returns
    -------
    dataset : Dataset
        Standardized battery dataset.

    Warnings
    --------
    UserWarning
        If `extra_columns` are not found in the source data or conflict with any
        of the standardized headers. Also, if no valid headers are found and an
        empty dataset is returned.

    See Also
    --------
    HeaderAliases : Customize the column/header mapping for standardization.

    Notes
    -----
    By default, only aliases of Seconds, Amps, Volts, Cycle, Step, State, Ah,
    Wh, and DateTime are included. If you'd like to ensure that additional data
    columns are included, use the `extra_columns` parameter.

    Examples
    --------
    The following example shows how to read in data from a `.csv` file using a
    few of the available options.

    .. code-block:: python

        import ampworks as amp

        # read in the file using all default options
        data = amp.read_csv('data.csv')

        # specify custom aliases for a couple column headers
        aliases = amp.HeaderAliases(Seconds='Time_s', Amps='Current_A')
        data = amp.read_csv('data.csv', aliases=aliases)

        # include extra columns for temperature and notes
        extra_cols = {'Temperature': float, 'Notes': None}
        data = amp.read_csv('data.csv', extra_columns=extra_cols)

    """
    from ampworks import Dataset, HeaderAliases
    from ampworks._checks import _check_type
    from ampworks._core._headers import (
        standardize_headers, header_matches, REQUIRED_HEADERS,
    )

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

        warn(f"No valid aliases found for {REQUIRED_HEADERS} in {filepath}."
             " Returning empty dataset.")

    return Dataset()
