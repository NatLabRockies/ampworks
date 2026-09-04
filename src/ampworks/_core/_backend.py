from __future__ import annotations

import re

from typing import Set, Dict, Sequence, Callable

import pandas as pd

AliasSet = Set[str] | Sequence[str]
AliasMap = Dict[str, float | Callable[[float], float] | None]

__all__ = [
    'AliasSet',
    'AliasMap',
    '_build_default_alias',
    '_strip_chars',
    '_astype_float',
    '_find_alias_match',
]


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

    # standardize casing and translate or remove common separators
    convert = '[(/,_'
    transmap = str.maketrans(convert, '.' * len(convert), ' -#<>)]')
    text = string.lower().translate(transmap)

    # collapse multiple dots
    text = re.sub(r"\.+", ".", text)

    return text.strip('.')  # remove trailing period, if present


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


def _find_alias_match(
    norm_raw: Dict[str, str],
    alias: AliasMap | AliasSet,
) -> str | None:
    """
    If norm_raw (a dictionary of normalized to raw headers, from source data)
    contains a match for the normalized header (i.e., norm) and the given alias,
    return the associated alias name/key and raw header. Otherwise, return None.

    """

    def _is_match(norm_head: str, alias_key: str) -> bool:
        # directly compare if key is just name or norm_head is assumed name.unit
        if '.' not in alias_key:
            return norm_head.replace('.', '') == alias_key

        if norm_head.count('.') == 1:
            return norm_head == alias_key

        # split at dot and look for a match in the normalized head segments
        name, units = alias_key.split('.')
        norm_seg = norm_head.split('.')

        is_match = any(
            ''.join(norm_seg[:i]) == name and ''.join(norm_seg[i:]) == units
            for i in range(1, len(norm_seg))
        )

        return is_match

    for norm_head, raw_head in norm_raw.items():
        alias_key = next((k for k in alias if _is_match(norm_head, k)), None)
        if alias_key is not None:
            return alias_key, raw_head

    return None
