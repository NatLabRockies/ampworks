from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scipy.optimize import minimize
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    from ampworks.dqdv import DqdvSpline


def match_peaks(
    charge: DqdvSpline,
    discharge: DqdvSpline,
    x0: float | None = None,
    display: bool = False,
) -> tuple[float, DqdvSpline]:
    """
    Find symmetric `iR` shift to align dQdV peaks.

    This function finds the optimal symmetric `iR` shift to apply to charge and
    discharge dQdV curves to align their peaks. The shifted curves are then
    averaged to produce a final dQdV curve. In addition to the `iR` value, a new
    `DqdvSpline` object is returned containing the averaged dQdV curve. Use this
    function to estimate an open-circuit voltage curve from low-rate charge and
    discharge data.

    Parameters
    ----------
    charge : DqdvSpline
        A fitted dQdV spline object for low-rate charge data.
    discharge : DqdvSpline
        A fitted dQdV spline object for low-rate discharge data.
    x0 : float | None, optional
        Initial guess for the symmetric iR shift, by default None, which uses a
        heuristic based on peak locations.
    display : bool, optional
        Whether to display intermediate optimization results, by default False.

    Returns
    -------
    iR, spline : tuple[float, DqdvSpline]
        The optimal symmetric iR shift and the averaged dQdV spline.

    See Also
    --------
    ~ampworks.dqdv.DqdvSpline : Smoothing spline for dQdV curves.

    """
    import ampworks as amp

    # good starting guess
    soc = np.linspace(0, 1, 201)
    chg = {'Volts': charge.volts_(soc), 'dqdv': charge.dqdv_(soc)}
    dis = {'Volts': discharge.volts_(soc), 'dqdv': discharge.dqdv_(soc)}
    if x0 is None:
        idx1 = np.argmax(np.abs(chg['dqdv']))
        idx2 = np.argmax(np.abs(dis['dqdv']))

        x0 = 0.5 * (chg['Volts'][idx1] - dis['Volts'][idx2])

    # interpolate dqdv expressions at common voltages
    Vmin = 0.5 * (np.min(chg['Volts']) + np.min(dis['Volts']))
    Vmax = 0.5 * (np.max(chg['Volts']) + np.max(dis['Volts']))

    n = np.ceil((Vmax - Vmin) / 1e-3)
    volts = np.linspace(Vmin, Vmax, n.astype(int))

    def shift_curves(iR: float) -> tuple[np.ndarray, np.ndarray]:
        """Symmetric shift of charge and discharge curves by iR."""
        dqdv_chg = np.interp(volts, chg['Volts'] - iR, chg['dqdv'])
        dqdv_dis = np.interp(volts, dis['Volts'] + iR, dis['dqdv'])
        return dqdv_chg, dqdv_dis

    # error function
    def errfn(x: float) -> float:
        """Sum of squared differences between shifted curves."""
        dqdv_chg, dqdv_dis = shift_curves(x)
        return np.linalg.norm(dqdv_chg - dqdv_dis)

    bounds = (0., 0.5*np.trapezoid(chg['Volts'] - dis['Volts'], x=soc))

    def callback(intermediate_result) -> None:
        """Print intermediate results, both x and function value."""
        return print(intermediate_result)

    result = minimize(errfn, x0, method='L-BFGS-B', bounds=[bounds],
                      callback=callback if display else None)

    # final approximation by averaging derivatives
    dqdv_chg, dqdv_dis = shift_curves(result.x)

    dqdv = 0.5 * (dqdv_chg + dqdv_dis)

    soc = cumulative_trapezoid(dqdv, volts, initial=0.)  # re-normalize
    soc = soc / np.max(np.abs(soc))

    # fake current and time to construct DqdvSpline output
    amps = np.ones(volts.size)
    seconds = soc / amps * 3600.

    data = amp.Dataset({'Seconds': seconds, 'Amps': amps, 'Volts': volts})
    spline = amp.dqdv.DqdvSpline().fit(data)

    return result.x.item(), spline
