from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:  # pragma: no cover
    from ._tables import DegModeTable, DqdvFitTable


def calc_lam_lli(fit_table: DqdvFitTable) -> pd.DataFrame:
    r"""
    Calculate degradation modes.

    Uses full cell capacity and fitted x0/x1 values from dqdv/dvdq fits to
    calculate theoretical electrode capacities, loss of active material (LAM),
    and loss of lithium inventory (LLI). The calculations are summarized below.
    For more detail and discussion, please refer to [1]_.

    Electrode capacities (Q) and loss of active material (LAM) are

    .. math::

        Q_{ed} = \frac{\rm capacity}{x_{1,ed} - x_{0,ed}}, \quad \quad
        {\rm LAM}_{ed} = 1 - \frac{Q_{ed}}{Q_{ed}[0]},

    where :math:`ed` is generic for 'electrode'. Outputs use 'n' and 'p' to
    differentiate between negative and positive electrodes, respectively. Loss
    of lithium inventory (LLI) is

    .. math::

        {\rm Inv} = x_{0,n}Q_{n} + (1 - x_{0,p})Q_{p}, \quad \quad
        {\rm LLI} = 1 - \frac{\rm Inv}{\rm Inv[0]},

    where :math:`{\rm Inv}` is the total lithium inventories using capacities
    :math:`Q` from above. If standard deviations of the x0/x1 stoichiometries
    are available in `results` (and are not NaN), then they are propagated
    to give uncertainty estimates for the LAM/LLI values. Reported uncertainties
    come from first-order Taylor series assumptions. If you trust your x0/x1
    fits but see large or inconsistent uncertainties then it is also safe to
    trust the LAM/LLI values, but you may want to neglect LAM/LLI uncertainties.
    Note that `(1 - xp0)` is used instead of just `xp0` because x0 refers
    to the delithiated state of the positive electrode whereas `xn0` refers
    to the lithiated state of the negative electrode, and does not require the
    same inversion.

    Parameters
    ----------
    fit_table : DqdvFitTable
        Table containing rows for fitted x0/x1 values from dqdv/dvdq fits.

    Returns
    -------
    deg_table : DegModeTable
        Electrode capacities (Q) and loss of active material (LAM) for the
        negative (n) and positive (p) electrodes, and loss of lithium inventory
        (LLI). Capacities are in Ah. All other outputs are unitless.

    See Also
    --------
    ~ampworks.dqdv.DqdvFitter : Access to fitting routines.
    ~ampworks.dqdv.DegModeTable : Table of calculated degradation modes.

    References
    ----------
    .. [1] A. Weng, J. B. Siegel, and A. Stefanopoulou, "Differential voltage
       analysis for battery manufacturing process control," Frontiers in Energy
       Research, 2023, DOI: 10.3389/fenrg.2023.1087269

    """
    from ampworks.dqdv._tables import DegModeTable

    df = fit_table.df.copy()
    extra_cols = fit_table._extra_cols

    Ah = df.Ah.to_numpy()

    xn0, xn0_std = df.xn0.to_numpy(), df.xn0_std.to_numpy()
    xn1, xn1_std = df.xn1.to_numpy(), df.xn1_std.to_numpy()
    xp0, xp0_std = df.xp0.to_numpy(), df.xp0_std.to_numpy()
    xp1, xp1_std = df.xp1.to_numpy(), df.xp1_std.to_numpy()

    Qn = Ah / (xn1 - xn0)
    Qp = Ah / (xp1 - xp0)

    dQn = Ah / (xn1 - xn0)**2  # ignore lead -1 for xn1 b/c **2 below
    Qn_std = np.sqrt((dQn*xn1_std)**2 + (dQn*xn0_std)**2)

    dQp = Ah / (xp1 - xp0)**2  # ignore lead -1 for xp1 b/c **2 below
    Qp_std = np.sqrt((dQp*xp1_std)**2 + (dQp*xp0_std)**2)

    LAMn = 1. - Qn / Qn[0]
    LAMp = 1. - Qp / Qp[0]

    LAMn_std = Qn_std / Qn[0]
    LAMp_std = Qp_std / Qp[0]

    inv = xn0*Qn + (1. - xp0)*Qp
    LLI = 1. - inv / inv[0]

    inv_std = np.sqrt(
        ((Qn + xn0*dQn)*xn0_std)**2           # contribution from xn0
        + ((xn0*dQn)*xn1_std)**2                # contribution from xn1
        + ((-Qp + (1. - xp0)*dQp)*xp0_std)**2   # contribution from xp0
        + (((1. - xp0)*dQp)*xn1_std)**2         # contribution from xp1
    )

    LLI_std = inv_std / inv[0]

    aging = pd.DataFrame({
        'Qn': Qn, 'Qn_std': Qn_std,
        'Qp': Qp, 'Qp_std': Qp_std,
        'Qc': Ah,
        'LAMn': LAMn, 'LAMn_std': LAMn_std,
        'LAMp': LAMp, 'LAMp_std': LAMp_std,
        'LLI': LLI, 'LLI_std': LLI_std,
    })

    for col in extra_cols:
        aging[col] = df[col]

    return DegModeTable(aging)


def plot_lam_lli(deg_table: DegModeTable, x_col: str | None = None,
                 std: bool = False) -> None:
    """
    Plot degradation modes.

    Parameters
    ----------
    deg_table : DegModeTable
        Container holding calculated degradation modes (LAM and LLI).
    x_col : str | None, optional
        A column name from 'fit_table` to use for the x-axis. If None (default)
        then the row indices are used.
    std : bool, optional
        Include shaded regions for estimated standard deviations of the LAM and
        LLI values when True. Default is False.

    See Also
    --------
    ~ampworks.dqdv.DqdvFitter : Access to the fitting routines.
    ~ampworks.dqdv.DegModeTable : Table of calculated degradation modes.
    ~ampworks.dqdv.calc_lam_lli : Calculate degradation modes before plottting.

    """
    from ampworks.utils import _ExitHandler
    from ampworks.plotutils import format_ticks

    df = deg_table.df.copy()

    if x_col is None:
        xplt, xlabel = df.index, 'Index'
    else:
        xplt, xlabel = df[x_col], x_col.capitalize()

    shaded = {'alpha': 0.2, 'color': 'C0'}

    _, axs = plt.subplots(
        2, 3, figsize=[9.0, 3.75], sharex=True, constrained_layout=True,
    )

    df.plot(
        x_col, ['Qn', 'Qp', 'Qc', 'LAMn', 'LAMp', 'LLI'], subplots=True,
        color='C0', legend=False, xlabel=xlabel, ax=axs.flatten(),
    )

    # first row: Qn, Qp, Qc
    if std:
        Qn, Qn_std = df[['Qn', 'Qn_std']].T.to_numpy()
        axs[0, 0].fill_between(xplt, Qn - Qn_std, Qn + Qn_std, **shaded)

        Qp, Qp_std = df[['Qp', 'Qp_std']].T.to_numpy()
        axs[0, 1].fill_between(xplt, Qp - Qp_std, Qp + Qp_std, **shaded)

    axs[0, 0].set_ylabel(r'$Q_{\rm NE}$ [Ah]')
    axs[0, 1].set_ylabel(r'$Q_{\rm PE}$ [Ah]')
    axs[0, 2].set_ylabel(r'$Q_{\rm cell}$ [Ah]')

    # second row: LAMn, LAMp, LLI
    if std:
        LAM, LAM_std = df[['LAMn', 'LAMn_std']].T.to_numpy()
        axs[1, 0].fill_between(xplt, LAM - LAM_std, LAM + LAM_std, **shaded)

        LAM, LAM_std = df[['LAMp', 'LAMp_std']].T.to_numpy()
        axs[1, 1].fill_between(xplt, LAM - LAM_std, LAM + LAM_std, **shaded)

        LLI, LLI_std = df[['LLI', 'LLI_std']].T.to_numpy()
        axs[1, 2].fill_between(xplt, LLI - LLI_std, LLI + LLI_std, **shaded)

    axs[1, 0].set_ylabel(r'LAM$_{\rm NE}$ [$-$]')
    axs[1, 1].set_ylabel(r'LAM$_{\rm PE}$ [$-$]')
    axs[1, 2].set_ylabel(r'LLI [$-$]')

    # formatting
    format_ticks(axs, xdiv=2, ydiv=2)

    _ExitHandler.register_atexit(plt.show)
