from __future__ import annotations

from typing import TYPE_CHECKING

from ampworks import _checks as _chk

if TYPE_CHECKING:
    from ampworks import Dataset


def add_power(
    data: Dataset,
    *,
    col_name: str = 'Watts',
    amps_alias: str = 'Amps',
    volts_alias: str = 'Volts',
) -> Dataset:
    """
    Add a power column to a dataset.

    Calculates power as current times voltage, preserving the input's sign
    convention (e.g., negative during discharge if current is negative).

    Parameters
    ----------
    data : Dataset
        The input dataset.
    col_name : str, optional
        Name of the column to add, by default 'Watts'.
    amps_alias : str, optional
        Name of the column containing current in amps, by default 'Amps'.
    volts_alias : str, optional
        Name of the column containing voltage in volts, by default 'Volts'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a power column.

    Examples
    --------
    Below we add a power column to a dataset using default settings.

    >>> data = amp.Dataset(...)
    >>> ds = add_power(data)

    """
    _chk._check_columns(data, [amps_alias, volts_alias])

    ds = data.copy()
    ds[col_name] = ds[amps_alias] * ds[volts_alias]
    return ds


def add_c_rate(
    data: Dataset,
    *,
    amps_1c: float,
    col_name: str = 'CRate',
    amps_alias: str = 'Amps',
) -> Dataset:
    """
    Add a C-rate column to a dataset.

    Calculates C-rate as current divided by the magnitude of the 1C current,
    preserving the input's sign convention (e.g., negative during discharge if
    current is negative).

    Parameters
    ----------
    data : Dataset
        The input dataset.
    amps_1c : float
        The 1C current, in amps, used to calculate C-rate.
    col_name : str, optional
        Name of the column to add, by default 'CRate'.
    amps_alias : str, optional
        Name of the column containing current in amps, by default 'Amps'.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a C-rate column.

    Examples
    --------
    Below we add a C-rate column to a dataset using default settings and a 1C
    current of 1.0 A.

    >>> data = amp.Dataset(...)
    >>> ds = add_c_rate(data, amps_1c=1.0)

    If your analysis requires a positive definition of C-rate, regardless of
    the sign of the current, use `ds['CRate'].abs()` in downstream processes.

    """
    _chk._check_columns(data, [amps_alias])

    ds = data.copy()
    ds[col_name] = ds[amps_alias] / abs(amps_1c)
    return ds
