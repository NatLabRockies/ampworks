from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    import ampworks as amp

__all__ = [
    '_infer_state',
    '_calc_soc',
    '_calc_relative_time',
]


def _infer_state(data: amp.Dataset) -> None:
    """
    Assign a 'State' column to *data* based on the sign of 'Amps'.

    Values are set to `'C'` (charge), `'D'` (discharge), and `'R'` (rest) where
    current is positive, negative, and zero, respectively. The column is written
    in-place.

    Parameters
    ----------
    data : Dataset
        Dataset with an 'Amps' column.

    """
    data['State'] = 'R'
    data.loc[data['Amps'] > 0, 'State'] = 'C'
    data.loc[data['Amps'] < 0, 'State'] = 'D'


def _calc_soc(data: amp.Dataset, charging: bool | None = None) -> None:
    """
    Compute state of charge and write it to `data['SOC']` in-place.

    SOC is estimated by integrating `Amps` over time using the trapezoidal rule.
    For charging, the result is normalised so that SOC runs from 0 to 1; for
    discharging, it runs from 1 to 0.

    Parameters
    ----------
    data : Dataset
        DataFrame with 'Amps' and 'Seconds' columns.
    charging : bool or None, optional
        `True` if the dataset represents a net-charge direction (SOC increases);
        `False` for a net-discharge direction (SOC decreases). The default is
        `None`, in which case the direction is inferred from the data (i.e., a
        net charge has an increase in voltage from start to end, and discharge
        has the opposite trend).

    Notes
    -----
    This function assumes that the data starts at a known SOC of either 0 or 1,
    depending on if the net protocol charges or discharges. Furthermore, it
    assumes that the maximum accumulated capacity corresponds to a full charge
    or discharge. If these assumptions are not met, SOC values may be incorrect.

    Since this function uses a trapezoidal integration to estimate SOC, it may
    be inaccurate if the data is very sparse or noisy. In such cases, consider
    using an alternate method, possibly using the `Ah` column directly, if one
    is available.

    """
    Ah = cumulative_trapezoid(data['Amps'], data['Seconds'] / 3600., initial=0.)

    increasing_volts = (data['Volts'].iloc[-1] > data['Volts'].iloc[0])
    charging = increasing_volts if charging is None else charging

    if charging:
        data['SOC'] = Ah / Ah.max()
    else:
        data['SOC'] = 1. - Ah / Ah.min()


def _calc_relative_time(
    data: amp.Dataset,
    groupby_cols: str | Sequence[str],
    col_name: str = 'RelativeTime',
) -> None:
    """
    Compute relative time within each group and write it to *data* in-place.

    For every row, the result is `Seconds - Seconds.iloc[0]` within the group
    defined by *groupby_cols*. Groups can be cycles, steps, or a more complex
    combination of a cycle/state, etc. so each segment has a zero-referenced
    time in the new column.

    Parameters
    ----------
    data : Dataset
        DataFrame with a 'Seconds' column, and columns needed for grouping.
    groupby_cols : str or Sequence[str]
        Column names to group by before computing relative time.
    col_name : str, optional
        Name of the output column. Defaults to `'RelativeTime'`.

    """
    groups = data.groupby(groupby_cols)
    data[col_name] = groups['Seconds'].transform(lambda x: x - x.iloc[0])
