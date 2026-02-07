from __future__ import annotations

from warnings import warn
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset


def _plot_pulses(data: Dataset, **fig_kw) -> None:
    """
    Plot voltage profile with detected pulses highlighted.

    Parameters
    ----------
    data : Dataset
        Input Dataset with 'Hours', 'Amps', 'Volts', 'DisPulse', 'ChgPulse',
        and 'StepTime'columns. These can all be added using _detect_pulses().
    **fig_kw : dict, optional
        Additional keyword arguments to use when plotting. A full list of names,
        types, descriptions, and defaults is given below.
    figsize : (int, int) or None, optional
        Figure size (width, height) in pixels. Set either dimension to None for
        responsiveness. In Jupyter, only width can resize; height is fixed. The
        default is (800, 500).
    save : str or None, optional
        Path to save the plot as HTML. If not in a Jupyter notebook and save is
        None, a temporary file is still created and is opened in the browser.

    """
    from ampworks.plotutils._style import PLOTLY_TEMPLATE
    from ampworks.plotutils._render import _render_plotly

    save = fig_kw.get('save', None)
    figsize = fig_kw.get('figsize', (800, 500))

    # Two-row subplot with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.3, 0.7], vertical_spacing=0.05,
    )

    # Detect appropriate units for current
    max_current = data['Amps'].abs().max()
    if max_current >= 1e-1:
        ycol, yunits = 'Amps', 'A'
    elif max_current >= 1e-4:
        ycol, yunits = 'milliAmps', 'mA'
        data[ycol] = data['Amps'] * 1e3
    else:
        ycol, yunits = 'microAmps', 'uA'
        data[ycol] = data['Amps'] * 1e6

    # Full current and voltage traces
    current = px.line(data, x='Hours', y=ycol)
    current.data[0].update(line_color='black')

    voltage = px.line(data, x='Hours', y='Volts')
    voltage.data[0].update(line_color='black')

    fig.add_trace(current.data[0], row=1, col=1)
    fig.add_trace(voltage.data[0], row=2, col=1)

    fig.update_xaxes(title_text='Hours', row=2, col=1)
    fig.update_yaxes(title_text=ycol, row=1, col=1)
    fig.update_yaxes(title_text='Volts', row=2, col=1)

    # Add shaded pulses and markers
    for pulse, color in zip(['DisPulse', 'ChgPulse'], ['red', 'blue']):

        for idx, g in data.groupby(pulse, dropna=True):
            t0, t1 = g['Hours'].iloc[0], g['Hours'].iloc[-1]
            i0, i1 = g[ycol].iloc[0], g[ycol].iloc[-1]
            v0, v1 = g['Volts'].iloc[0], g['Volts'].iloc[-1]

            trel0, trel1 = g['StepTime'].iloc[0], g['StepTime'].iloc[-1]

            hover_amps = (
                f"StepTime: %{{customdata:.3f}} s<br>"
                f"Current: %{{y:.3f}} {yunits}"
            )
            hover_volts = (
                "StepTime: %{customdata:.3f} s<br>"
                "Voltage: %{y:.3f} V"
            )
            customdata = [trel0, trel1]

            fig.add_vrect(
                x0=t0, x1=t1,  row=1, col=1,
                fillcolor=color, opacity=0.3, line_width=0,
            )
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[i0, i1], name=pulse + f"{idx}",
                mode='markers', marker=dict(color=color, size=8),
                hovertemplate=hover_amps, customdata=customdata,
            ), row=1, col=1)

            fig.add_vrect(
                x0=t0, x1=t1, row=2, col=1,
                fillcolor=color, opacity=0.3, line_width=0,
            )
            fig.add_trace(go.Scatter(
                x=[t0, t1], y=[v0, v1], name=pulse + f"{idx}",
                mode='markers', marker=dict(color=color, size=8),
                hovertemplate=hover_volts, customdata=customdata,
            ), row=2, col=1)

    # Adjust layout and styling; then save and display
    fig.update_layout(template=PLOTLY_TEMPLATE, showlegend=False)
    _render_plotly(fig=fig, figsize=figsize, save=save)


def _detect_pulses(
    data: Dataset,
    tmin: float = 0.,
    tmax: float = 20.,
    steps: list[int] | None = None,
    plot: bool = False,
    **fig_kw,
) -> Dataset:
    """
    Detect charge and discharge pulses in the input Dataset. Optionally plot
    the voltage profile with detected pulses highlighted.

    Parameters
    ----------
    data : Dataset
        Input Dataset with 'Seconds', 'Volts', and 'Amps' columns.
    tmin : float, optional
        Minimum pulse duration in seconds, by default 0. Any non-rest segments
        shorter than this will be ignored.
    tmax : float, optional
        Maximum pulse duration in seconds, by default 20. Any non-rest segments
        longer than this will be ignored.
    steps : list[int] or None, optional
        Explicit list of step numbers associated with HPPC pulses. If None,
        pulses are autodetected based on state transitions. Requires a 'Step'
        column in 'data'. Defaults to None.
    plot : bool, optional
        Whether to plot the current and voltage profiles with detected pulses
        highlighted, by default False.
    **fig_kw : dict, optional
        Additional keyword arguments to use when plotting. A full list of names,
        types, descriptions, and defaults is given below.
    figsize : (int, int) or None, optional
        Figure size (width, height) in pixels. Set either dimension to None for
        responsiveness. In Jupyter, only width can resize; height is fixed. The
        default is (800, 500).
    save : str or None, optional
        Path to save the plot as HTML. If not in a Jupyter notebook and save is
        None, a temporary file is still created and is opened in the browser.

    Returns
    -------
    data : amp.Dataset
        A copy of the input Dataset with additional columns: 'Hours', 'State',
        'Ah', 'SOC', 'Segment', 'StepTime', 'DisPulse', 'ChgPulse'.

    """
    df = data.copy()
    df = df.reset_index(drop=True)

    df['Seconds'] -= df['Seconds'].min()
    df['Hours'] = df['Seconds'] / 3600.

    # Create State column
    df['State'] = 'R'
    df.loc[df['Amps'] > 0, 'State'] = 'C'
    df.loc[df['Amps'] < 0, 'State'] = 'D'

    # Add Ah and SOC columns
    is_net_charge = df['Volts'].iloc[0] < df['Volts'].iloc[-1]
    sign = +1 if is_net_charge else -1

    df['Ah'] = cumulative_trapezoid(sign*df['Amps'], df['Hours'], initial=0.)

    if is_net_charge:
        df['SOC'] = df['Ah'] / df['Ah'].max()
    else:
        df['SOC'] = 1. - df['Ah'] / df['Ah'].max()

    # Create 'Step' column to group by State and Step
    shifted_state = df['State'].shift(fill_value=df['State'].iloc[0])
    df['Segment'] = (df['State'] != shifted_state).cumsum()

    groups = df.groupby(['State', 'Segment'])
    df['StepTime'] = np.nan

    # Loop over (State, Step) groups to locate charge/discharge pulses
    df['DisPulse'] = pd.NA
    df['ChgPulse'] = pd.NA

    dis_count = 1
    chg_count = 1

    for (state, _), g in groups:

        idx = g.index
        if idx[0] != df.index[0]:
            idx = np.hstack([idx[0] - 1, idx], dtype=int)

        steptime = df.loc[idx, 'Seconds'] - df.loc[idx[0], 'Seconds']

        before, after = idx[0], idx[-1] + 1
        if (state == 'R') or (steptime.max() > tmax):
            continue
        elif any(df.loc[[before, after], 'State'] != 'R'):
            continue
        elif (steps is not None) and (g['Step'].unique()[0] not in steps):
            continue

        if (state == 'D') and (steptime.max() >= tmin):
            df.loc[idx, 'StepTime'] = steptime
            df.loc[idx, 'DisPulse'] = dis_count
            dis_count += 1
        elif (state == 'C') and (steptime.max() >= tmin):
            df.loc[idx, 'StepTime'] = steptime
            df.loc[idx, 'ChgPulse'] = chg_count
            chg_count += 1

    # Plot, if requested
    if plot:
        _plot_pulses(df, **fig_kw)

    return df


def extract_impedance(
    data: Dataset,
    tmin: float = 0.,
    tmax: float = 20.,
    sample_times: list[float] | None = None,
    steps: list[int] | None = None,
    area: float | None = None,
    plot: bool = False,
    **fig_kw,
) -> pd.DataFrame:
    """
    Extract impedance from HPPC data.

    HPPC, or hybrid pulse power characterization, is a common protocol used to
    measure the power capability and internal resistance. The protocol consists
    of a series of charge and/or discharge pulses performed at various states of
    charge. This function extracts the impedance during these pulses. The pulses
    can either be autodetected based on state transitions from rest and non-rest
    periods, or explicitly specified using step numbers. By default, only the
    instantaneous and end-of-pulse impedance values are reported, but additional
    sample times can be specified in the input. The "instantaneous" impedance is
    defined as the impedance calculated using the first sample after the pulse
    start. The "end" impedance is calculated using the last sample before the
    pulse end.

    Area specific impedance (ASI) is also calculated if the electrode or cell
    area is provided. Otherwise, the output will only provide the absolute
    impedance in Ohms. ASI, when provided, is reported in Ohms-cm2. Make sure
    the area is in cm2 to ensure correct units.

    There are many ways to write an HPPC protocol. If you're looking for a place
    to start, consider reading through the recommendations provided by Idaho
    National Labs [1]_. Or, consider using the HPPC protocol provided below,
    which was used for testing the algorithm. Note that the algorithm assumes
    that a cell starts at 100% SOC.

    1. Rest for 60 minutes
    2. Discharge at C/3 for 18 minutes (~10% SOC)
    3. Rest for 60 minutes
    4. Discharge at 1C for 30 seconds
    5. Rest for 40 seconds
    6. Charge at 0.75C for 10 seconds
    7. Rest for 40 seconds
    8. Repeat steps 2-7 until 10% SOC is reached
    9. Discharge at C/3 until lower voltage limit
    10. Rest for 60 minutes

    This protocol extracts impedance at nine SOC points, roughly between 10% and
    90%. This is to make sure that the protocol does not exceed upper or lower
    voltage limits during the pulse as the cell nears 100% or 0% SOC. Note that
    the pulse currents and durations are different in the charge and discharge
    directions. This is not a requirement, but was convenient for testing the
    algorithm. In the example dataset, samples were logged every 0.1 seconds for
    steps 4-7 above, and every 10 seconds for all other steps.

    Parameters
    ----------
    data : Dataset
        The sliced HPPC data to process. Must have, at a minimum, the columns
        for `{'Seconds', 'Volts', 'Amps'}`. See notes for more information.
    tmin : float, optional
        Minimum pulse duration in seconds, by default 0.
    tmax : float, optional
        Maximum pulse duration in seconds, by default 20.
    sample_times : list[float] or None, optional
        Relative times to sample each pulse's impedance, in seconds. If None
        (default), only "instantaneous" and end of pulse impedance are reported.
        `NaN` is used for sample times that cannot be interpolated.
    steps : list[int] or None, optional
        Explicit list of step numbers associated with HPPC pulses. If None,
        pulses are autodetected based on state transitions. Requires a `Step`
        column in `data`. Defaults to None.
    area : float, optional
        Electrode or cell area in cm2. Used to calculate area specific impedance
        (ASI) in Ohms-cm2. If None (default), ASI values are not reported.
    plot : bool, optional
        Whether to plot the current and voltage profiles with detected pulses
        highlighted, by default False.
    **fig_kw : dict, optional
        Additional keyword arguments to use when plotting. A full list of names,
        types, descriptions, and defaults is given below.
    figsize : (int, int) or None, optional
        Figure size (width, height) in pixels. Set either dimension to None for
        responsiveness. In Jupyter, only width can resize; height is fixed. The
        default is (800, 500).
    save : str or None, optional
        Path to save the plot as HTML. If not in a Jupyter notebook and save is
        None, a temporary file is still created and is opened in the browser.

    Returns
    -------
    impedance : pd.DataFrame
        Impedance table with 'PulseNum', 'State', 'Hours_0', 'SOC_0', 'AmpsAvg',
        and 'StepTime_i', 'Volts_i', 'Ohms_i', and 'ASI_i' columns. Index i
        marks: 0=pre-pulse, 1=instantaneous, 2..k=sample times, N=end. Note that
        ASI is only reported if 'area' is provided.

    Notes
    -----
    Rests within the dataset are expected to have a current exactly equal to
    zero. The autodetection routine for pulses relies on this to identify pulse
    start and end times. If you need to zero out currents below a threshold, you
    can use `data.zero_below('Amps', threshold)` before calling this function.
    Furthermore, even when using explicit 'steps', the algorithm checks for rest
    periods before and after each pulse. If these are not present, the pulse is
    ignored.

    This algorithm expects charge/discharge currents to be positive/negative,
    respectively. If your sign convention is the opposite, 'SOC_0' values in the
    output will be incorrect. Furthermore, the algorithm assumes the protocol
    starts at either 0% or 100% SOC and ends at the opposite extreme. If this is
    not the case, the 'SOC_0' values in the output will be incorrect.

    References
    ----------
    .. [1] J. Christophersen, "Battery Test Manual For Electric Vehicles,
       Revision 3," OSTI, 2015, DOI: 10.2172/1186745

    Examples
    --------
    >>> import seaborn as sns
    >>> data = amp.datasets.load_datasets('hppc/hppc_discharge')
    >>> impedance = amp.hppc.extract_impedance(data, tmax=31, sample_times=[5])
    >>> ax = sns.scatterplot(data=impedance, x='SOC_0', y='Ohms_2', hue='State')
    >>> ax.set_xlabel('SOC [-]')
    >>> ax.set_ylabel('R5 [Ohms]')
    >>> print(impedance)

    """
    # Validate required columns
    required = ['Seconds', 'Amps', 'Volts']
    if not all(col in data.columns for col in required):
        raise ValueError(f"'data' is missing columns, {required=}.")

    # Ensure consistency when 'steps' is provided
    if (steps is not None) and ('Step' not in data.columns):
        raise ValueError("'data' requires 'Step' column to use 'steps' input.")

    elif steps is not None:

        missing = set(steps) - set(data['Step'].unique())
        if missing:
            raise ValueError(f"'steps' has values not in 'data': {missing=}.")

        ignore = data[data['Step'].isin(steps)]
        ignore = ignore.loc[ignore['Amps'] == 0, 'Step'].unique().tolist()
        if ignore:
            warn(f"Ignoring 'steps' with rest state: {ignore=}")

    # Detect pulses using 'steps' or state transitions
    df = _detect_pulses(
        data, tmin=tmin, tmax=tmax, steps=steps, plot=plot, **fig_kw,
    )

    # Calculate and store impedance for detected pulses at 'sample_times'
    if sample_times is None:
        sample_times = []

    sample_times = sorted(sample_times)

    impedance = None
    dis_groups = df.groupby('DisPulse', dropna=True)
    chg_groups = df.groupby('ChgPulse', dropna=True)
    for state, groups in zip(['D', 'C'], [dis_groups, chg_groups]):

        for (idx, g) in groups:

            soc0 = g['SOC'].iloc[0]
            seconds0 = g['Seconds'].iloc[0]
            amps_avg = g.loc[g['Amps'] != 0, 'Amps'].mean()

            # Build list of step times to sample impedance:
            # 0 -> just before pulse
            # 1 -> instant after pulse start
            # 2..k -> requested sample times
            # N -> end of pulse
            steptimes = g['StepTime'].iloc[[0, 1]].to_list()
            steptimes += sample_times
            steptimes.append(g['StepTime'].iloc[-1])

            volts = np.interp(
                steptimes,
                g['StepTime'],
                g['Volts'],
                left=np.nan,
                right=np.nan,
            )

            resist = np.abs(volts - volts[0]) / np.abs(amps_avg)

            asi_dict = {}
            if area is not None:
                asi = resist * area
                asi_dict = {f"ASI_{i}": [v] for i, v in enumerate(asi)}

            row = {
                'PulseNum': [idx],
                'State': [state],
                'Hours_0': [seconds0 / 3600.],
                'SOC_0': [soc0],
                'AmpsAvg': [amps_avg],
            }

            row.update({f"StepTime_{i}": [v] for i, v in enumerate(steptimes)})
            row.update({f"Volts_{i}": [v] for i, v in enumerate(volts)})
            row.update({f"Ohms_{i}": [v] for i, v in enumerate(resist)})
            row.update(asi_dict)

            impedance = pd.concat(
                [impedance, pd.DataFrame(row)], ignore_index=True,
            )

    rename = {}
    N = len(steptimes) - 1
    for name in ['StepTime', 'Volts', 'Ohms', 'ASI']:
        rename[f"{name}_{N}"] = f"{name}_N"

    impedance.rename(columns=rename, inplace=True)

    return impedance
