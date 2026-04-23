from __future__ import annotations

import textwrap

from warnings import warn
from typing import TYPE_CHECKING, Generator, Sequence

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset


def format_alias(names: Sequence[str], units: Sequence[str]) -> list[str]:
    """
    Build alias strings from names and units.

    Parameters
    ----------
    names : Sequence[str]
        Base signal names.
    units : Sequence[str]
        Unit labels used in the source files.

    Returns
    -------
    aliases : list[str]
        Alias strings containing unit-only, name-only, and name.unit forms.

    """
    aliases = list(units)

    for name in names:
        aliases.append(name)
        for unit in units:
            aliases.append(f"{name}.{unit}")

    return aliases


def strip_chars(string: str | list[str] | None) -> str | list[str] | None:
    """
    Normalize header text for matching.

    Parameters
    ----------
    string : str or list[str] or None
        Header text to normalize.

    Returns
    -------
    stripped : str or list[str] or None
        Lowercased text with common separators removed.

    """
    if string is None:
        return None
    if isinstance(string, list):
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


class HeaderAliases:
    """Header alias definitions."""

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
        """
        A container that allows users to specify custom header aliases for their
        data. These are used to automatically find and standardize columns when
        loading data. `ampworks` uses default aliases for any headers that are
        not provided here.

        Parameters
        ----------
        Seconds : str or list[str] or None, optional
            Aliases for the standardized Seconds column.
        Amps : str or list[str] or None, optional
            Aliases for the standardized Amps column.
        Volts : str or list[str] or None, optional
            Aliases for the standardized Volts column.
        Cycle : str or list[str] or None, optional
            Aliases for the standardized Cycle column.
        Step : str or list[str] or None, optional
            Aliases for the standardized Step column.
        State : str or list[str] or None, optional
            Aliases for the standardized State column.
        Ah : str or list[str] or None, optional
            Aliases for the standardized Ah column.
        Wh : str or list[str] or None, optional
            Aliases for the standardized Wh column.
        DateTime : str or list[str] or None, optional
            Aliases for the standardized DateTime column.

        """
        from ampworks._checks import _check_inner_type, _check_type

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

    def __getitem__(self, key: str) -> list[str]:
        """Return aliases for a standardized header name."""
        if key in self.__slots__:
            return getattr(self, key)
        raise KeyError(f"{key} not found in {self.__class__.__name__}")

    def __repr__(self) -> str:  # pragma: no cover
        data = {k: v for k, v in self.items()}
        summary = "\n".join([f"{k}={v!r}," for k, v in data.items()])
        summary = textwrap.indent(summary, " " * 4)
        return f"{self.__class__.__name__}(\n{summary}\n)"

    def keys(self) -> list[str]:
        """Return standardized header names supported by the alias set."""
        return list(self.__slots__)

    def items(self) -> Generator[tuple[str, list[str]], None, None]:
        """Iterate over `(std_header, aliases)` pairs."""
        for slot in self.__slots__:
            yield (slot, getattr(self, slot))


def header_matches(
    headers: list[str],
    targets: list[str],
    aliases: HeaderAliases,
) -> bool:
    """
    Check headers for required targets.

    Parameters
    ----------
    headers : list[str]
        Source headers to evaluate.
    targets : list[str]
        Standardized target names that must be present.
    aliases : HeaderAliases
        Alias definitions used for matching.

    Returns
    -------
    checks : bool
        True when all target headers are matched.

    """
    normalized = strip_chars(headers)

    checks = {}
    for key in targets:
        checks[key] = any(alias in normalized for alias in aliases[key])

    return all(checks.values())


def standardize_headers(
    data: pd.DataFrame,
    aliases: HeaderAliases | None = None,
    extra_columns: dict[str, type | None] | None = None,
) -> Dataset:
    """
    Map source columns to `ampworks` standards.

    Parameters
    ----------
    data : pandas.DataFrame
        Source data frame with raw cycler headers.
    aliases : HeaderAliases or None, optional
        Alias mapping used to identify standardized columns. If None, defaults
        are used.
    extra_columns : dict[str, type or None] or None, optional
        Extra source columns to keep in output using exact source names as
        keys. Values define cast type. Use None to keep inferred dtype.

    Returns
    -------
    dataset : Dataset
        Standardized dataset.

    Warnings
    --------
    UserWarning
        Raised when standardized aliases are missing, requested extra columns
        are not found, or requested extra columns conflict with standardized
        output columns.

    """
    from ampworks import Dataset

    if aliases is None:
        aliases = HeaderAliases()

    df = Dataset()

    unit_factors = {
        'Amps': {
            ('ma', 'mamps', 'milliamps'): 0.001,
        },
        'Ah': {
            ('mah', 'mahr', 'mamphr'): 0.001,
        },
        'Seconds': {
            ('min', 'mins', 'minute', 'minutes'): 60.0,
            ('h', 'hr', 'hrs', 'hour', 'hours'): 3600.0,
        },
    }

    # Match as-imported headers with standardized headers
    for std_header in aliases.keys():
        for raw_header in data.columns:
            normalized = strip_chars(raw_header)
            if normalized not in aliases[std_header]:
                continue

            if std_header not in df.columns:
                df[std_header] = data[raw_header]

            # Standardize units
            if std_header in unit_factors:
                for units, factor in unit_factors[std_header].items():
                    if any(unit in normalized for unit in units):
                        df[std_header] = df[std_header].astype(float) * factor
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

        sign = df['State'].map({'R': 0.0, 'C': 1.0, 'D': -1.0}).fillna(1.0)
        df['Amps'] = sign * df['Amps'].abs()

    # Create 'Ah' and 'Wh' from separate charge and discharge columns
    if any(header not in df.columns for header in ['Ah', 'Wh']):
        ah_headers = ['charge' + header for header in aliases['Ah']]
        wh_headers = ['charge' + header for header in aliases['Wh']]
        for raw_header in data.columns:
            normalized = strip_chars(raw_header)
            if normalized in ah_headers:
                df['Ah'] = data[raw_header]
                discharge_ah = data[raw_header.replace('Charge', 'Discharge')]
                df.loc[df['State'] == 'D', 'Ah'] = discharge_ah
            if normalized in wh_headers:
                df['Wh'] = data[raw_header]
                discharge_wh = data[raw_header.replace('Charge', 'Discharge')]
                df.loc[df['State'] == 'D', 'Wh'] = discharge_wh

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

    # Keep user-requested non-standardized columns from source data
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
                 f"{set(data.columns)}.")

        if skipped_extra:
            warn(f"Skipped some conflicting 'extra_columns': {skipped_extra=}."
                 f" Existing are {set(df.columns)}.")

    return df
