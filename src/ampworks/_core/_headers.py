from __future__ import annotations

import textwrap

from warnings import warn
from typing import TYPE_CHECKING, Set, Dict, Sequence, Callable

import pandas as pd

if TYPE_CHECKING:
    from ampworks import Dataset

AliasSet = Set[str] | Sequence[str]
AliasMap = Dict[str, float | Callable[[float], float] | None]


# construct HEADER_ALIASES dictionary from base names and unit->factor maps
def _build_default_alias(
    names: Set[str],
    units: Dict[str, float] | None,
) -> AliasSet | AliasMap:
    """
    Build a default alias set or map from names and units.

    Parameters
    ----------
    names : Set[str]
        Base alias names.
    units : Dict[str, float] or None
        Mapping of unit label (key) and conversion factor (value) to the base
        unit (i.e., after standardization). If None, only build AliasSet.

    Returns
    -------
    aliases : AliasSet or AliasMap
        Alias set or map for given names and units. If units is not None, the
        map keys include name only, unit only, and name.unit variants.

    """
    if units is None:
        return set(names)

    aliases = dict(units)
    for name in names:
        aliases[name] = 1.0
        for unit, factor in units.items():
            aliases[f"{name}.{unit}"] = factor

    return aliases


TIME_NAMES = {'t', 'time', 'test', 'testtime', 'totaltime'}
TIME_UNITS = {
    's': 1.0,
    'sec': 1.0,
    'seconds': 1.0,
    'min': 60.0,
    'mins': 60.0,
    'minute': 60.0,
    'minutes': 60.0,
    'h': 3600.0,
    'hr': 3600.0,
    'hrs': 3600.0,
    'hour': 3600.0,
    'hours': 3600.0,
}

CURRENT_NAMES = {'i', 'amperage', 'current'}
CURRENT_UNITS = {
    'a': 1.0,
    'amps': 1.0,
    'ma': 1e-3,
    'mamps': 1e-3,
    'milliamps': 1e-3,
}

VOLTAGE_NAMES = {'voltage', 'potential', 'ecell'}
VOLTAGE_UNITS = {'v': 1.0, 'volts': 1.0}

STATE_NAMES = {'state', 'md', 'mode'}
STEP_NAMES = {'step', 'ns', 'stepindex'}
CYCLE_NAMES = {'cycle', 'cyc', 'cycleindex', 'cyclenumber', 'cyclec', 'cyclep'}

CAPACITY_NAMES = {'capacity', 'amphours', 'cap'}
CAPACITY_UNITS = {
    'ah': 1.0,
    'ahr': 1.0,
    'amphr': 1.0,
    'mah': 1e-3,
    'mahr': 1e-3,
    'mamphr': 1e-3,
}

ENERGY_NAMES = {'energy', 'watthours', 'ener'}
ENERGY_UNITS = {'wh': 1.0, 'whr': 1.0, 'watthr': 1.0, 'uwatthr': 1e-6}

DATETIME_NAMES = {'datetime', 'dpttime', 'realtime'}

DEFAULT_ALIASES: Dict[str, AliasSet | AliasMap] = {
    'Seconds': _build_default_alias(TIME_NAMES, TIME_UNITS),
    'Amps': _build_default_alias(CURRENT_NAMES, CURRENT_UNITS),
    'Volts': _build_default_alias(VOLTAGE_NAMES, VOLTAGE_UNITS),
    'Cycle': _build_default_alias(CYCLE_NAMES, None),
    'Step': _build_default_alias(STEP_NAMES, None),
    'State': _build_default_alias(STATE_NAMES, None),
    'Ah': _build_default_alias(CAPACITY_NAMES, CAPACITY_UNITS),
    'Wh': _build_default_alias(ENERGY_NAMES, ENERGY_UNITS),
    'DateTime': _build_default_alias(DATETIME_NAMES, None),
}

REQUIRED_HEADERS = {'Seconds', 'Amps', 'Volts'}


class HeaderAliases:
    """Header alias definitions."""

    __slots__ = ('Seconds', 'Amps', 'Volts', 'Cycle', 'Step', 'State', 'Ah',
                 'Wh', 'DateTime')

    def __init__(
        self,
        *,
        Seconds: AliasMap | None = None,
        Amps: AliasMap | None = None,
        Volts: AliasMap | None = None,
        Cycle: AliasSet = None,
        Step: AliasSet = None,
        State: AliasSet = None,
        Ah: AliasMap | None = None,
        Wh: AliasMap | None = None,
        DateTime: AliasSet | None = None,
        extend_defaults: bool | Sequence[str] = False,
    ) -> None:
        """
        A container to hold header alias and conversion definitions to get data
        into a standard format. Provide your own aliases, or use defaults.

        Parameters
        ----------
        Seconds : AliasMap or None, optional
            Time column aliases and converters. None uses internal defaults.
        Amps : AliasMap or None, optional
            Current column aliases and converters. None uses internal defaults.
        Volts : AliasMap or None, optional
            Voltage column aliases and converters. None uses internal defaults.
        Cycle : AliasSet or None, optional
            Cycle column aliases. None uses internal defaults.
        Step : AliasSet or None, optional
            Step column aliases. None uses internal defaults.
        State : AliasSet or None, optional
            State column aliases. None uses internal defaults.
        Ah : AliasMap or None, optional
            Capacity column aliases and converters. None uses internal defaults.
        Wh : AliasMap or None, optional
            Energy column aliases and converters. None uses internal defaults.
        DateTime : AliasSet or None, optional
            DateTime column aliases. None uses internal defaults.
        extend_defaults : bool or Sequence[str], optional
            How to augment or override default aliases. True extends defaults.
            False (default) replaces defaults with provided values. To extend
            some while replacing others, supply a list of which field names to
            extend, others will be replaced, where the field names are the
            attribute names of the class (e.g., 'Seconds', 'Amps', etc.).

        Notes
        -----
        All aliases default to `None`, which uses internal defaults. When not
        using the defaults, provided aliases must be given an `AliasMap` for any
        fields which may require unit conversions, or an `AliasSet` for fields
        which don't require unit conversions.

        An `AliasMap` is just a dictionary where the keys are the alias names
        and values are converters. A converter can be a float if the conversion
        is just a multiplicative factor, or a callable like `f(float) -> float`
        if the conversion is more complex. When no conversion is needed, the
        value can be either `None` or `1.0`.

        An `AliasSet` is used for a few of the fields which don't require unit
        conversion (e.g., Cycle, Step, and State). For these fields, simply
        provide a list of the alias names. Even if you want to enforce only one
        alias, it must still be provided as a list of one item.

        Examples
        --------
        The following examples show how to use `HeaderAliases` to specify custom
        aliases. Any inputs that are skipped will use a list of defaults. Be
        aware that all parameters must be provided as keywords to avoid improper
        ordering.

        Plain aliases can use `None` if unit conversion is not needed:

        >>> import ampworks as amp
        >>> aliases = amp.HeaderAliases(
        ...     Seconds={'elapsed_s': None},
        ...     Amps={'current_amps': None, 'current_a': None},
        ... )

        If a column's units don't match the standardized unit, use the value to
        specify the conversion factor or function. Factors (float) multiply the
        source value to convert to the standard unit. For example, `s = 60*min`:

        >>> aliases = amp.HeaderAliases(Seconds={'elapsed_min': 60.0})

        Converters can also be arbitrary callables like `f(float) -> float`:

        >>> aliases = amp.HeaderAliases(Volts={'v_mv': lambda x: x / 1000})

        Use `extend_defaults` to add to the built-in defaults instead of fully
        replacing them, either for all provided fields or for select fields:

        >>> aliases = amp.HeaderAliases(
        ...     Seconds={'elapsed_s': None},
        ...     extend_defaults=True,
        ... )
        >>> aliases = amp.HeaderAliases(
        ...     Seconds={'elapsed_s': None},
        ...     Amps={'current_a': None},
        ...     extend_defaults=['Seconds'],
        ... )

        All of the above examples focus on values that may or may not need to be
        converted to new units. There are also fields which never require unit
        conversion, which are given as a list of strings instead of by a map:

        >>> aliases = amp.HeaderAliases(
        ...     Cycle=['CycleNumber'],
        ...     Step=['StepIndex', 'StepNumber'],
        ...     extend_defaults=True,
        ... )

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

        _check_type('extend_defaults', extend_defaults, (bool, Sequence))
        if isinstance(extend_defaults, Sequence):
            _check_inner_type('extend_defaults', extend_defaults, str)
            extend_fields = set(extend_defaults)
        else:
            extend_fields = set(params) if extend_defaults else set()

        invalid = extend_fields - set(params)
        if invalid:
            raise ValueError(
                f"'extend_defaults' has invalid field(s) {invalid}. Expected a"
                f" subset of {list(params.keys())}.",
            )

        # loop over fields and add to class instance
        for name, value in params.items():
            extend = name in extend_fields
            setattr(self, name, _format_user_alias(name, value, extend))

    def __getitem__(self, key: str) -> AliasMap | AliasSet:  # noqa: E501
        """Return the alias map or set for a standardized header name."""
        if key in self.__slots__:
            return getattr(self, key)
        raise KeyError(f"{key} not found in {type(self).__name__}")

    def __repr__(self) -> str:
        data = {k: self[k] for k in self.keys()}
        summary = "\n".join([f"{k}={v!r}," for k, v in data.items()])
        summary = textwrap.indent(summary, " " * 4)
        return f"{type(self).__name__}(\n{summary}\n)"

    def keys(self) -> list[str]:
        """Return standardized header names supported by the alias set."""
        return list(self.__slots__)


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
    normalized = _strip_chars(headers)

    checks = {}
    for key in targets:
        checks[key] = any(alias in normalized for alias in aliases[key])

    return all(checks.values())


def standardize_headers(
    data: pd.DataFrame,
    aliases: HeaderAliases | None = None,
    extra_columns: Dict[str, type | None] | None = None,
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
    extra_columns : Dict[str, type or None] or None, optional
        Extra source columns to keep in output using exact source names as
        keys. Values define cast type. Use None to keep inferred dtype.

    Returns
    -------
    data : Dataset
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

    # Match as-imported headers with standardized headers
    for std_header in aliases.keys():
        for raw_header in data.columns:

            # Store column if there is a match, and doesn't already exist
            normalized = _strip_chars(raw_header)
            if normalized not in aliases[std_header]:
                continue
            if std_header in df.columns:
                continue

            df[std_header] = data[raw_header]

            # Standardize units using the alias's converter, if any
            if not isinstance(aliases[std_header], dict):
                continue

            converter = aliases[std_header][normalized]
            df[std_header] = _astype_float(df[std_header])
            if (converter is None) or (converter == 1.0):
                continue
            elif callable(converter):
                df[std_header] = df[std_header].apply(converter)
            else:
                df[std_header] = df[std_header] * converter

    # Create 'State' data if not present
    if ('State' not in df.columns) and ('Amps' in df.columns):
        df['Amps'] = _astype_float(df['Amps'])

        df['State'] = 'R'
        df.loc[df['Amps'] > 0, 'State'] = 'C'
        df.loc[df['Amps'] < 0, 'State'] = 'D'

    # Guarantee sign 'Amps' sign convention (+ charge, - discharge)
    if 'State' in df.columns:
        rename_bitrode = {'REST': 'R', 'DCHG': 'D', 'CHRG': 'C'}
        df['State'] = df['State'].replace(rename_bitrode)

        df['Amps'] = _astype_float(df['Amps'])
        df['State'] = df['State'].astype(str)

        sign = df['State'].map({'R': 0.0, 'C': 1.0, 'D': -1.0}).fillna(1.0)
        df['Amps'] = sign * df['Amps'].abs()

    # Create 'Ah' and 'Wh' from separate charge and discharge columns
    if any(header not in df.columns for header in ['Ah', 'Wh']):
        ah_headers = ['charge' + header for header in aliases['Ah']]
        wh_headers = ['charge' + header for header in aliases['Wh']]
        for raw_header in data.columns:
            normalized = _strip_chars(raw_header)
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
                df[std_header] = _astype_float(df[std_header])
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


def _format_user_alias(
    std_header: str,
    alias: AliasSet | AliasMap | None,
    extend_defaults: bool,
):
    """
    Format user-provided aliases (AliasMap or AliasSet types) to be used in the
    `standardize_headers` function.

    Parameters
    ----------
    std_header : str
        One of the standard header alias names, from DEFAULT_ALIASES keys.
    alias : AliasSet or AliasMap or None
        User-provided alias mapping or set. If None, defaults are used.
    extend_defaults : bool
        Whether or not to extend the current alias's defaults. If not extended,
        the user-provided values replace the internal defaults.

    Returns
    -------
    formatted : AliasMap or AliasSet
        The formatted (and optionally extended) alias for `std_header`.

    """
    from ampworks._checks import _check_type, _check_inner_type

    defaults = DEFAULT_ALIASES[std_header].copy()
    if alias is None:
        return defaults

    # handle AliasSet options
    if std_header in ['Cycle', 'Step', 'State', 'DateTime']:
        _check_type(f"{std_header}", alias, (Sequence, None))
        if isinstance(alias, str):
            raise TypeError(
                f"{std_header} alias must be Sequence[str], but got str."
            )

        _check_inner_type(f"{std_header}", alias, str)
        alias = _strip_chars(alias)

        formatted = set(alias)
        if extend_defaults:
            formatted.update(defaults)

        return formatted

    # handle AliasMap options
    _check_type(f"{std_header}", alias, (dict, None))
    _check_inner_type(f"{std_header}", alias.keys(), str)

    formatted = {_strip_chars(k): v for k, v in alias.items()}
    for k, v in formatted.items():
        if (v is None) or callable(v):
            continue

        formatted[k] = float(v)

    if extend_defaults:
        formatted = {**defaults, **formatted}  # adopt user's if duplicate keys

    return formatted


def _strip_chars(string: str | list[str] | None) -> str | list[str] | None:
    """
    Normalize header aliases text for matching.

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
        return [_strip_chars(s) for s in string]

    transmap = str.maketrans('[(/,', '....', ' _-#<>)]')
    return string.lower().translate(transmap).replace('..', '.').strip('.')


def _astype_float(series: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to float, coercing errors.

    If numeric, ensures a float. Else, convert to a string, strip commas and
    hash symbols, then coerce to numeric. Non-convertible values become NaN.

    Parameters
    ----------
    series : pd.Series
        Input Series to convert.

    Returns
    -------
    series : pd.Series
        Converted Series with float dtype.

    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    series = series.astype(str).replace('[,#]', '', regex=True)
    return pd.to_numeric(series, errors='coerce').astype(float)
