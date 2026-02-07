from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scipy.stats import linregress
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset


def extract_params(data: Dataset, radius: float, tmin: float = 1,
                   tmax: float = 60, return_all: bool = False) -> pd.DataFrame:
    """
    Extracts parameters from GITT data.

    GITT, or galvanostatic intermittent titration technique, is an experiment
    that applies intermittent low-rate charge or discharge pulses separated by
    long rest periods that establish equilibrium. The experiments can be used
    to extract important parameters for physics-based models. For example, the
    open-circuit voltage and solid-phase diffusivity.

    The following protocol was used to test this algorithm:

    1. Rest for 5 min, log data every 10 s.

    2. Charge (or discharge) at C/20 for 11 min; include a voltage limit. Log
       data every 0.2 s or every 5 mV change.

    3. Rest for 135 min, log data every 10 min or every 5 mV change.

    4. Stop if voltage limit reached in (2), otherwise repeat (2) and (3).

    The protocol assumes formation cycles have already been completed and that
    the cell was rested until equilibrium before starting the steps above.
    Implementation details are available in [1]_.

    Parameters
    ----------
    data : Dataset
        The sliced GITT data to process. Must have, at a minimum, columns for
        `{'Seconds', 'Amps', 'Volts'}`. See notes for more information.
    radius : float
        The representative particle radius of your active material (in meters).
        It's common to use D50 / 2, i.e., the median radius of a distribution.
    tmin : float, optional
        The minimum relative pulse time (in seconds) to use when fitting sqrt(t)
        vs. voltage for time constants. Default is 1.
    tmax : float, optional
        The maximum relative pulse time (in seconds) to use when fitting sqrt(t)
        vs. voltage for time constants. Default is 60. See notes for more info.
    return_all : bool, optional
        If False (default), only the extracted parameters vs. state of charge
        are returned. If True, also returns stats with info about each pulse.

    Returns
    -------
    params : pd.DataFrame
        Table of parameters. Columns include 'SOC' (state of charge, -), 'Ds'
        (diffusivity, m2/s), and 'Eeq' (equilibrium potential, V).
    stats : pd.DataFrame
        Only returned if `return_all=True`. Provides additional stats about
        each pulse, including errors from the sqrt(t) vs. voltage regressions.

    Raises
    ------
    ValueError
        'data' is missing columns, required=['Seconds', 'Amps', 'Volts'].
    ValueError
        'data' should not include both charge and discharge segments.

    Notes
    -----
    Rests within the dataset are expected to have a current exactly equal to
    zero. You can use `data.zero_below('Amps', threshold)` to manually zero
    out currents below some tolerance, if needed. This should be done prior to
    passing in the dataset to this function.

    This algorithm expects charge/discharge currents to be positive/negative,
    respectfully. If your sign convention is the opposite, the mapping to 'SOC'
    in the output will be reversed. You must process data in one direction at
    a time. In other words, if you performed the GITT protocol in both charge
    and discharge directions you should slice your data into two datasets and
    call this routine twice.

    The algorithm assumes that `sqrt(t)` vs. voltage is approximately linear.
    Mathematically this occurs on time scales much less than the time constant
    `tau = R**2 / D`. Large `tmax` that violate `tmax << tau` will have
    incorrect results. See the references for a more detailed discussion. Also,
    if a pulse has fewer than two data points between the set relative `tmin`
    and `tmax` then the linear regression performed to find the diffusivity
    and equilbrium potential will return `NaN` for both.

    References
    ----------
    .. [1] Z. Geng, Y. Chien, M. J. Lacey, T. Thiringer, and D. Brandell,
       "Validity of solid-state Li+ diffusion coefficient estimation by
       electrochemical approaches for lithium-ion batteries," EA, 2022,
       DOI: 10.1016/j.electacta.2021.139727

    Examples
    --------
    >>> data = amp.datasets.load_datasets('gitt/gitt_discharge')
    >>> params, stats = amp.gitt.extract_params(data, 1.8e-6, return_all=True)
    >>> params.plot('SOC', 'Eeq')
    >>> params.plot('SOC', 'Ds', logy=True)
    >>> print(params)
    >>> print(stats)

    """
    required = ['Seconds', 'Amps', 'Volts']
    if not all(col in data.columns for col in required):
        raise ValueError(f"'data' is missing columns, {required=}.")

    charging = any(data['Amps'] > 0.)
    discharging = any(data['Amps'] < 0.)

    if charging and discharging:
        raise ValueError(
            "'data' should not include both charge and discharge segments."
        )

    df = data.copy()
    df = df.reset_index(drop=True)

    # States based on current direction: charge, discharge, or rests
    df['State'] = 'R'
    df.loc[df['Amps'] > 0, 'State'] = 'C'
    df.loc[df['Amps'] < 0, 'State'] = 'D'

    # Add in state-of-charge column to map each value to an SOC
    Ah = cumulative_trapezoid(
        df['Amps'].abs(), df['Seconds'] / 3600, initial=0,
    )

    if charging:
        df['SOC'] = Ah / Ah.max()
    elif discharging:
        df['SOC'] = 1 - Ah / Ah.max()

    # Count each time a rest/charge or rest/discharge changeover occurs
    pulse = (df['State'] != 'R') & (df['State'].shift(fill_value='R') == 'R')
    df['Pulse'] = pulse.cumsum()

    # Relative time of each rest/charge or rest/discharge step
    groups = df.groupby(['Pulse', 'State'])
    df['StepTime'] = groups['Seconds'].transform(lambda x: x - x.iloc[0])

    # Remove last cycle if not complete, i.e., ended on charge or discharge
    if df.iloc[-1]['State'] != 'R':
        df = df[df['Pulse'] != df['Pulse'].max()].reset_index(drop=True)

    # Record summary stats for each loop, immediately before the pulses
    groups = df[df['State'] != 'R'].groupby('Pulse', as_index=False)
    summary = groups.agg(lambda x: x.iloc[0])

    # Store slope and intercepts (V = m*t^0.5 + b) for each pulse
    groups = df.groupby('Pulse')

    regression = None
    for idx, g in groups:

        if idx > 0:

            rest = g[g['State'] == 'R']
            pulse = g[g['State'] != 'R']

            dt_rest = rest['StepTime'].max() - rest['StepTime'].min()
            dt_pulse = pulse['StepTime'].max() - pulse['StepTime'].min()

            pulse = pulse[
                (pulse['StepTime'] >= tmin) &
                (pulse['StepTime'] <= tmax)
            ]

            x = np.sqrt(pulse['StepTime'])
            y = pulse['Volts']

            if len(x) <= 1:
                x, y = [0, 1], [np.nan, np.nan]

            result = linregress(x, y)
            new_row = pd.DataFrame({
                'Pulse': [idx],
                'Eeq': [result.intercept],
                'Eeq_err': [result.intercept_stderr],
                'dUdrt': [result.slope],
                'dUdrt_err': [result.stderr],
                'dt_rest': [dt_rest],
                'dt_pulse': [dt_pulse],
            })

            regression = pd.concat([regression, new_row], ignore_index=True)

    stats = pd.merge(summary, regression, on='Pulse')
    stats['dEdt'] = np.gradient(stats['Volts'], np.cumsum(stats['dt_pulse']))

    params = pd.DataFrame({
        'SOC': stats['SOC'],
        'Ds': 4./9./np.pi * (radius * stats['dEdt']/stats['dUdrt'])**2,
        'Eeq': stats['Eeq'],
    })

    params.sort_values(by='SOC', inplace=True, ignore_index=True)

    if return_all:
        return params, stats

    return params
