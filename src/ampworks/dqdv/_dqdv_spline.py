from __future__ import annotations
from typing import Self, Literal, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import make_splrep
from scipy.integrate import cumulative_trapezoid

if TYPE_CHECKING:  # pragma: no cover
    import ampworks as amp
    import numpy.typing as npt

CapacityMethod = Literal['auto', 'provided', 'integrated']


class DqdvSpline:
    """Smoothing spline for dQdV curves."""

    def __init__(self, capacity_method: CapacityMethod = 'auto') -> None:
        r"""
        A class for fitting and evaluating smoothing splines for dQdV data. Use
        the `fit` method to fit splines to datasets. The API takes inspiration
        from scikit-learn's estimator interface. Instance variables (see list
        below) and all methods ending with an underscore (e.g., `volts_(soc)`,
        `score_`) are only available after fitting.

        The smoothing spline is only applied to the voltage vs. state of charge
        (SOC) data. dVdQ and dQdV curves are then computed from the derivative,
        and its inverse, of the voltage spline. All splines are parameterized by
        SOC, including dQdV. So to plot dQdV vs. voltage, first evaluate the
        voltage spline at the desired SOC values, then evaluate the dQdV spline
        at the same SOC values.

        Parameters
        ----------
        capacity_method : {'auto', 'provided', 'integrated'}, optional
            How capacity is determined. 'auto' (default) uses the 'Ah' column if
            present, otherwise integrates current over time. 'provided' requires
            and uses 'Ah', throwing errors if missing. 'integrated' forces the
            integration of current and ignores 'Ah'.

        Attributes
        ----------
        Ah\_ : 1D np.array
            Stored capacities used in constructing the spline.
        SOC\_ : 1D np.array
            Stored state of charge values used in constructing the spline.
        Volts\_ : 1D np.array
            Stored voltage values used in constructing the spline, not to be
            confused with the spline itself, evaluated using `volts_()`.
        score\_ : float
            Root mean square error between smoothed and raw voltages.

        """
        options = ['auto', 'provided', 'integrated']
        if capacity_method not in options:
            raise ValueError(
                f"'capacity_method' expected a value in {options} but received"
                f" {capacity_method}."
            )

        self._capacity_method = capacity_method

    def fit(self, data: amp.Dataset, s: float = 0.) -> Self:
        """
        Fit a smoothing spline to voltage vs. SOC data, and use the spline to
        construct dVdQ and dQdV splines using the derivative and its inverse.

        Parameters
        ----------
        data : amp.Dataset
            Sliced charge or discharge data to fit a spline to. Must contain,
            columns for `{'Ah', 'Volts'}` or `{'Seconds', 'Amps', 'Volts'}`
            depending on `capacity_method`. See notes for more information.
        s : float, optional
            The smoothing condition passed to SciPy's `make_splrep`. Controls
            the trade-off between closeness to the data and smoothness of fit
            according to

            .. code-block:: python

                sum( (g(x) - y)**2 ) <= s

            By default, `s=0.`, which results in an interpolating spline. Using
            a larger value produces a smoother spline. If you're unsure, start
            with a small value like `s=1e-4` and increase/decrease, as needed.

        Returns
        -------
        spline : Self
            The fitted `DqdvSpline` object.

        Raises
        ------
        ValueError
            Missing required columns in 'data'.
        ValueError
            Charge and discharge data must have positive and negative current,
            respectively. See notes for more information.
        ValueError
            Invalid 'Ah' column: minimum value is not zero and/or values are not
            monotonically increasing.

        Notes
        -----
        This method expects charge/discharge currencts to be positive/negative,
        respectively. If your sign convention is different, you will need to
        adjust the current data accordingly before fitting. Otherwise, internal
        checks will raise a `ValueError`. The sign convention does not change
        for half cells vs. full cells. Thus, NMC cathode half cells, graphite
        anode half cells, or any full cell should all have positive current when
        voltage is increasing (charging) and negative current when voltage is
        decreasing (discharging).

        The `capacity_method` parameter controls how capacity is determined and
        which required columns are expected in `data`. Capacity can either be
        taken directly from an 'Ah' column, or calculated by integrating current
        over time. If `capacity_method='provided'`, then the minimum required
        columns are {'Ah', 'Volts'}. If `capacity_method='integrated'`, then
        the minimum required columns are {'Seconds', 'Amps', 'Volts'}. Using the
        default `capacity_method='auto'` will accept either of these options,
        preferring the 'Ah' column if it is present, and integrating otherwise.

        In cases where an 'Ah' column is provided and used, the values will be
        checked to ensure they are valid. The minimum value must be zero and
        values must be monotonically increasing. Make sure your capacity column
        meets these requirements if you choose to use it.

        """
        data = data.reset_index(drop=True)

        # flag how to determine capacity
        if self._capacity_method == 'auto':
            use_Ah = 'Ah' in data.columns
        else:
            use_Ah = self._capacity_method == 'provided'

        required = {'Ah', 'Volts'} if use_Ah else {'Seconds', 'Amps', 'Volts'}
        if not required.issubset(data.columns):
            raise ValueError(
                f"Missing columns in 'data'. Expected at least {required} but"
                f" received {set(data.columns)}."
            )

        # integrate to get Ah column, if needed or requested
        is_net_charge = data['Volts'].iloc[0] < data['Volts'].iloc[-1]

        if not use_Ah:
            sign = +1 if is_net_charge else -1

            if is_net_charge and any(data['Amps'] <= 0.):
                raise ValueError("Charge data must have positive current.")
            if not is_net_charge and any(data['Amps'] >= 0.):
                raise ValueError("Discharge data must have negative current.")

            data['Ah'] = cumulative_trapezoid(
                sign*data['Amps'], x=data['Seconds'] / 3600., initial=0.,
            )

        # check Ah column for validity
        errors = []
        if not np.isclose(data['Ah'].min(), 0.0, atol=1e-12):
            errors.append("minimum value is not zero")

        if not data['Ah'].is_monotonic_increasing:
            errors.append("values are not monotonically increasing")

        if errors:
            raise ValueError("Invalid 'Ah' column: " + "; ".join(errors) + ".")

        # add SOC column, ensure increasing voltage with increasing SOC
        if is_net_charge:
            data['SOC'] = data['Ah'] / data['Ah'].max()
        else:
            data['SOC'] = 1. - data['Ah'] / data['Ah'].max()

        # fit smoothing spline
        _, mask = np.unique(data['SOC'], return_index=True)

        data = data.iloc[mask].reset_index(drop=True)
        data = data.sort_values('SOC', ignore_index=True)

        Ah = data['Ah'].to_numpy()
        SOC = data['SOC'].to_numpy()
        Volts = data['Volts'].to_numpy()

        # set spline and derivative
        self._volts = make_splrep(SOC, Volts, s=s)
        self._dvdq = self._volts.derivative()

        # add fit attributes
        self.Ah_ = Ah
        self.SOC_ = SOC
        self.Volts_ = Volts
        self.score_ = np.sqrt(np.mean((self.volts_(SOC) - Volts)**2))

        return self

    def plot(self) -> plt.Axes:
        """
        Plot the fitted splines against the original data.

        Returns
        -------
        axs : plt.Axes array
            A 2x2 axes object containing the plots.

        """
        from ampworks.utils import _ExitHandler
        from ampworks.plotutils import add_text, focused_limits, format_ticks

        volts_dat = self.Volts_
        dvdq_dat = np.gradient(self.Volts_, self.SOC_)
        dqdv_dat = 1. / dvdq_dat

        volts_fit = self.volts_(self.SOC_)
        dvdq_fit = self.dvdq_(self.SOC_)
        dqdv_fit = self.dqdv_(self.SOC_)

        mV_err = (volts_fit - volts_dat)*1e3

        _, axs = plt.subplots(2, 2, figsize=[8, 5], layout='tight')

        # voltage vs soc
        axs[0, 0].plot(self.SOC_, volts_dat, '.', color='C0', alpha=0.5)
        axs[0, 0].plot(self.SOC_, volts_fit, '--k')
        axs[0, 0].set_xlabel('SOC [-]')
        axs[0, 0].set_ylabel('Voltage [V]')
        axs[0, 0].legend(['Data', 'Spline'], frameon=False, loc='upper left')
        add_text(axs[0, 0], 0.6, 0.15, f"RMSE: {self.score_*1e3:.2f} mV")

        # voltage error vs soc
        axs[1, 0].plot(self.SOC_, mV_err, '-k')
        axs[1, 0].set_xlabel('SOC [-]')
        axs[1, 0].set_ylabel('Error [mV]')

        # dqdv vs voltage
        axs[0, 1].plot(volts_dat, dqdv_dat, '.', color='C0', alpha=0.5)
        axs[0, 1].plot(volts_fit, dqdv_fit, '--k')

        ymin, ymax = dqdv_fit.min(), dqdv_fit.max()
        ylim = (ymin - 0.05*(ymax - ymin), ymax + 0.05*(ymax - ymin))
        axs[0, 1].set_ylim(ylim)
        axs[0, 1].set_xlabel('dqdv [1/V]')
        axs[0, 1].set_ylabel('Voltage [V]')

        # dvdq vs soc
        axs[1, 1].plot(self.SOC_, dvdq_dat, '.', color='C0', alpha=0.5)
        axs[1, 1].plot(self.SOC_, dvdq_fit, '--k')

        ylim = focused_limits(dvdq_fit)
        axs[1, 1].set_ylim(ylim)
        axs[1, 1].set_xlabel('SOC [-]')
        axs[1, 1].set_ylabel('dvdq [V]')

        format_ticks(axs)

        _ExitHandler.register_atexit(plt.show)

    def volts_(self, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate the voltage spline at given SOC values.

        Parameters
        ----------
        soc : ArrayLike
            State of charge values at which to evaluate the voltage spline.

        Returns
        -------
        voltage : ArrayLike
            Voltage values corresponding to the given SOC values.

        Raises
        ------
        RuntimeError
            Call 'fit' before evaluating.

        """
        if not hasattr(self, '_volts'):
            raise RuntimeError("Call 'fit' before evaluating.")
        return self._volts(soc)

    def dvdq_(self, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate the dVdQ spline at given SOC values.

        Parameters
        ----------
        soc : ArrayLike
            State of charge values at which to evaluate the dVdQ spline.

        Returns
        -------
        dvdq : ArrayLike
            dVdQ values corresponding to the given SOC values. The output units
            are Volts instead of Volts per Ah, since splines are fit using SOC.

        Raises
        ------
        RuntimeError
            Call 'fit' before evaluating.

        """
        if not hasattr(self, '_dvdq'):
            raise RuntimeError("Call 'fit' before evaluating.")
        return self._dvdq(soc)

    def dqdv_(self, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate the dQdV spline at given SOC values.

        Parameters
        ----------
        soc : ArrayLike
            State of charge values at which to evaluate the dQdV spline.

        Returns
        -------
        dvdq : ArrayLike
            dQdV values corresponding to the given SOC values. The output units
            are 1/Volts instead of Ah/Volts, since splines are fit using SOC.

        Raises
        ------
        RuntimeError
            Call 'fit' before evaluating.

        """
        if not hasattr(self, '_dvdq'):
            raise RuntimeError("Call 'fit' before evaluating.")
        return 1. / self._dvdq(soc)
