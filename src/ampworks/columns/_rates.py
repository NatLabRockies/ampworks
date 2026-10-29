from __future__ import annotations

from ampworks import _checks as _chk
from ampworks._core._dataset import Dataset
from ampworks._core._std_names import AMPS, VOLTS, WATTS


def add_power(
    data: Dataset,
    *,
    output_col: str = WATTS,
    amps_col: str = AMPS,
    volts_col: str = VOLTS,
) -> Dataset:
    """
    Add a power column to a dataset.

    Calculates power as current times voltage, preserving the input's sign
    convention (e.g., negative during discharge if current is negative).

    Parameters
    ----------
    data : Dataset
        The input dataset.
    output_col : str, optional
        Name of power column to add; defaults to standard name.
    amps_col : str, optional
        Name of column containing current in amps; defaults to standard name.
    volts_col : str, optional
        Name of column containing voltage in volts; defaults to standard name.

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
    _chk._check_columns(data, [amps_col, volts_col])

    ds = Dataset(data)
    ds[output_col] = ds[amps_col] * ds[volts_col]
    return ds


def add_c_rate(
    data: Dataset,
    *,
    nominal_ah: float,
    output_col: str = 'CRate',
    amps_col: str = AMPS,
) -> Dataset:
    """
    Add a C-rate column to a dataset.

    Calculates C-rate as current divided by the magnitude of the 1C current,
    preserving the input's sign convention (e.g., negative during discharge if
    current is negative). The 1C current, in amps, is numerically equal to the
    nominal capacity, in ampere-hours.

    Parameters
    ----------
    data : Dataset
        The input dataset.
    nominal_ah : float
        The nominal capacity of the battery in ampere-hours, which defines the
        1C current used to calculate C-rate.
    output_col : str, optional
        Name of the column to add, by default 'CRate'.
    amps_col : str, optional
        Name of column containing current in amps; defaults to standard name.

    Returns
    -------
    ds : Dataset
        A modified copy of the input data, with a C-rate column.

    Raises
    ------
    ValueError
        If `nominal_ah` is zero.

    Examples
    --------
    Below we add a C-rate column to a dataset using default settings and a
    nominal capacity of 1.0 Ah (i.e., a 1C current of 1.0 A).

    >>> data = amp.Dataset(...)
    >>> ds = add_c_rate(data, nominal_ah=1.0)

    If your analysis requires a positive definition of C-rate, regardless of
    the sign of the current, use `ds['CRate'].abs()` in downstream processes.

    """
    _chk._check_columns(data, [amps_col])

    if nominal_ah == 0:
        raise ValueError("'nominal_ah' must be nonzero.")

    ds = Dataset(data)
    ds[output_col] = ds[amps_col] / abs(nominal_ah)
    return ds
