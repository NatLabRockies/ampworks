from __future__ import annotations

import warnings

from numbers import Real
from typing import Callable, Iterable, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
import scipy.interpolate as interp

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt

    from ampworks.utils import RichResult
    from ampworks.dqdv import DqdvFitResult


class DqdvFitter:

    def __init__(
        self, neg: pd.DataFrame = None, pos: pd.DataFrame = None,
        cell: pd.DataFrame = None, cost_terms: str | list[str] = 'all',
    ) -> None:
        """
        Wrapper for dQdV fitting.

        Fits electrode stoichiometries to replicate the voltage response of a
        full cell. Fitting relies on low-rate half-cell and full-cell charge
        and/or discharge curves to approximate open-circuit potentials.

        Internally, data is automatically used to generate smooth splines for
        the fitting routines. However, pre-processing to reduce noise in the
        measurements is recommended to avoid instabilities in the fits and
        derivative calculations.

        Note that reported stoichiometries x0/x1 are in reference to relative
        states of charge. Furthermore, the values differ in meaning between the
        negative and positive electrodes. xp0 and xp1 refer to the relative
        measure of how *delithiated* the positive electrode is, whereas xn0 and
        xn1 refer to how *lithiated* the negative electrode is. This concention
        is used so that x0 < x1 in both electrodes. Furthermore, it allows x0
        states to both refer to the SOC=0 state of the full cell and both x1
        values to the SOC=1 state of the full cell.

        Parameters
        ----------
        neg : pd.DataFrame
            Negative electrode OCV data.
        pos : pd.DataFrame
            Positive electrode OCV data.
        cell : pd.DataFrame
            Full cell OCV data.
        cost_terms : str or list[str], optional
            Error terms for optimization. 'all' (default) = ['voltage', 'dqdv',
            'dvdq']. Accepts a string (single term) or list (subset of terms).

        Notes
        -----
        * The dataframe inputs are all required. The default `None` values
          allow you to initialize the class first and add them one at a time.
          This is primarily to support interactions with the GUI.
        * When 'voltage' is included in `cost_terms`, an iR term is fit in
          addition to the x0/x1 stoichiometries. Otherwise, the ohmic iR offset
          is forced to 0. `cost_terms` can be modified after initialization
          via its property.

        """
        self._initialized = {}

        self.neg = neg
        self.pos = pos
        self.cell = cell

        self.cost_terms = cost_terms

    @property
    def neg(self) -> pd.DataFrame:
        """
        Get or set the negative electrode dataframe.

        Columns must include 'Ah' for capacity and 'Volts' for the half-cell
        voltage. All 'Ah' values must be positive, and there must be a zero
        reference somewhere in the column.

        """
        return self._neg

    @neg.setter
    def neg(self, value: pd.DataFrame) -> None:

        df = self._check_dataframe(value, 'neg')

        self._neg = None if df is None else df.copy()
        self._ocv_n, self._dvdq_n = self._build_splines(self._neg, 'neg')

    @property
    def pos(self) -> pd.DataFrame:
        """
        Get or set the positive electrode dataframe.

        Columns must include 'Ah' for capacity and 'Volts' for the half-cell
        voltage. All 'Ah' values must be positive, and there must be a zero
        reference somewhere in the column.

        """
        return self._pos

    @pos.setter
    def pos(self, value: pd.DataFrame) -> None:

        df = self._check_dataframe(value, 'pos')

        self._pos = None if df is None else df.copy()
        self._ocv_p, self._dvdq_p = self._build_splines(self._pos, 'pos')

    @property
    def cell(self) -> pd.DataFrame:
        """
        Get or set the full cell dataframe.

        Columns must include 'Ah' for capacity and 'Volts' for the full-cell
        voltage. All 'Ah' values must be positive, and there must be a zero
        reference somewhere in the column.

        """
        return self._cell

    @cell.setter
    def cell(self, value: pd.DataFrame) -> None:

        df = self._check_dataframe(value, 'cell')

        self._cell = None if df is None else df.copy()
        self._ocv_c, self._dvdq_c = self._build_splines(self._cell, 'cell')

        if self._initialized['cell']:
            self._soc = np.linspace(0., 1., 201)

            self._volt_data = self._ocv_c(self._soc)
            self._dvdq_data = self._dvdq_c(self._soc)
            self._dqdv_data = 1 / self._dvdq_data

    @property
    def cost_terms(self) -> list[str]:
        """
        Get or set which terms are included in the constrained fit's cost
        function. Options are 'voltage', 'dqdv', and/or 'dvdq'. You can also
        set to 'all' to conveniently select all three cost terms.

        """
        return self._cost_terms

    @cost_terms.setter
    def cost_terms(self, value: str | list[str]) -> None:

        options = ['voltage', 'dqdv', 'dvdq']

        if value == 'all':
            value = options.copy()
        elif isinstance(value, str):
            value = [value]

        if not isinstance(value, Sequence):
            raise TypeError("cost_terms must be a Sequence.")

        if len(value) == 0:
            raise ValueError("cost_terms is empty. Set to either 'all' or a"
                             f" subset of of {options}.")

        if not set(value).issubset(options):
            raise ValueError("cost_terms has at least one invalid value. It"
                             f" can only be 'all' or a subset of {options}.")

        self._cost_terms = value

    def _check_dataframe(self, df: pd.DataFrame, which: str) -> pd.DataFrame:
        """
        Verify that input dataframes have 'Ah' and 'Volts' columns.

        Parameters
        ----------
        df : pd.DataFrame
            Data to check for required 'Ah' and 'Volts' columns.
        which : {'neg', 'pos', 'cell'}
            Which splines to build. Used to track initialization.

        Returns
        -------
        df : None or pd.DataFrame
            None if dataset has not yet been initialized. Otherwise, pass the
            error checks and return the input DataFrame.

        Raises
        ------
        TypeError
            The 'df' input must be type pd.DataFrame.
        ValueError
            'df' is missing columns, required={'Ah', 'SOC', 'Volts'}.

        """
        self._initialized[which] = False

        required = {'SOC', 'Volts'}
        if which == 'cell':
            required.add('Ah')

        if df is None:
            pass
        elif not isinstance(df, pd.DataFrame):
            raise TypeError(f"'{which}' must be type pd.DataFrame.")
        elif not required.issubset(df.columns):
            raise ValueError(f"'{which}' is missing columns, {required=}.")

        return df

    def _build_splines(self, df: pd.DataFrame, which: str) -> Callable:
        """
        Generate OCV interpolation functions.

        Parameters
        ----------
        df : pd.DataFrame
            Data with 'SOC' and 'Volts' columns.
        which : {'neg', 'pos', 'cell'}
            Which splines to build. Used to track initialization.

        Returns
        -------
        ocv, dvdq : tuple[Callable]
            Spline interpolations for ocv and dvdq.

        """
        if df is None:
            return None, None

        _, mask = np.unique(df.SOC, return_index=True)

        df = df.iloc[mask].reset_index(drop=True)
        df = df.sort_values('SOC').reset_index(drop=True)

        # make sure neg has decreasing voltage with increasing SOC and pos/cell
        # have increasing so all splines are in reference to full-cell SOC.
        flip_soc = {
            'neg': lambda v0, v1: v0 < v1,
            'pos': lambda v0, v1: v0 > v1,
            'cell': lambda v0, v1: v0 > v1,
        }

        v0, v1 = df.Volts.iloc[0], df.Volts.iloc[-1]

        if flip_soc[which](v0, v1):
            df['SOC'] = 1.0 - df['SOC']

        df = df.sort_values('SOC').reset_index(drop=True)

        # build splines
        ocv = interp.make_splrep(df.SOC, df.Volts)
        dvdq = ocv.derivative()

        self._initialized[which] = True

        return ocv, dvdq

    def _check_initialized(self, func_name: str) -> None:
        """
        Check that the instance is fully initialized, will all splines and
        data for 'neg', 'pos', and 'cell'. If any is missing raise an error.

        Parameters
        ----------
        func_name : str
            Name of function performing check.

        Raises
        ------
        RuntimeError
            Can't run any functions until all data is available.

        """
        missing = [d for d, flag in self._initialized.items() if not flag]
        if missing:
            raise RuntimeError(f"Can't run '{func_name}' until all data is"
                               f" available. Missing {missing} data.")

    def _err_func(self, params: npt.ArrayLike) -> float:
        """
        The cost function for 'grid_search' and 'constrained_fit'.

        Parameters
        ----------
        params : ArrayLike, shape(n,)
            Array for xn0, xn1, xp0, xp1, and optionally iR.

        Returns
        -------
        err_total : float
            Total error based on a combination of cost_terms.

        """
        errs = self.err_terms(params)

        err_total = 0.  # faster when MAPE is fractional, so use (*1e-2) below
        if 'voltage' in self.cost_terms:
            err_total += errs['volt_err']*1e-2
        if 'dqdv' in self.cost_terms:
            err_total += errs['dqdv_err']*1e-2
        if 'dvdq' in self.cost_terms:
            err_total += errs['dvdq_err']*1e-2

        return err_total

    def get_ocv(self, which: str, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate an OCV spline.

        Parameters
        ----------
        which : {'neg', 'pos', 'cell'}
            Which OCV spline to evaluate.
        soc : ArrayLike
            Relative state-of-charge values, between [0, 1], to evaluate at.
            See notes for more information.

        Returns
        -------
        ocv : np.ndarray
            Evaluated OCV values.

        Raises
        ------
        ValueError
            'which' must be in ['neg', 'pos', 'cell'].
        RuntimeError
            If the requested spline has not yet been constructed.

        Notes
        -----
        Due to the internally storage of the half-cell data and splines, the
        OCV curves returned by this method are in opposite orders. Plotting the
        full [0, 1] window for the negative electrode will provide a curve with
        decreasing voltage from zero to 1 because x0/x1 refer to the state of
        how lithiated the negative electrode is. In contrast, the positive
        electrode OCV will return a curve with increasing potential because the
        fitted x0/x1 are in reference to how delithiated the positive electrode
        is. Evaluating the full cell curve also provides a curve with increasing
        voltage because the input is inpretted as the state of charge for the
        full cell. Values returned by all fitting routines are consistent with
        this orientation, and users can access the voltage windows of individual
        electrode potentials by passing the appropriate sections of the fitted
        x0/x1 values. For example, use `get_ocv(fit_result.x[0:2])` for the
        negative electrode and `get_ocv(fit_result[2:4])` for the positive
        electrode.

        """
        if which not in ['neg', 'pos', 'cell']:
            raise ValueError("'which' must be in ['neg', 'pos', 'cell'].")

        spline = getattr(self, f"_ocv_{which[0]}")
        if spline is None:
            raise RuntimeError(f"'{which}' splines are not constructed yet."
                               f" Set the '{which}' property first.")

        return spline(soc)

    def get_dvdq(self, which: str, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate a dvdq spline.

        Parameters
        ----------
        which : {'neg', 'pos', 'cell'}
            Which dvdq spline to evaluate.
        soc : ArrayLike
            Relative state-of-charge values, between [0, 1], to evaluate at.
            See notes for more information.

        Returns
        -------
        dvdq : np.ndarray
            Evaluated dvdq values.

        Raises
        ------
        ValueError
            'which' must be in ['neg', 'pos', 'cell'].
        RuntimeError
            If the requested spline has not yet been constructed.

        Notes
        -----
        The individual electrode datasets and splines are internally stored in
        opposite directions of one another. See the `get_ocv` method for more
        information to ensure you are evaluating each in the correct directions.

        """
        if which not in ['neg', 'pos', 'cell']:
            raise ValueError("'which' must be in ['neg', 'pos', 'cell'].")

        spline = getattr(self, f"_dvdq_{which[0]}")
        if spline is None:
            raise RuntimeError(f"'{which}' splines are not constructed yet."
                               f" Set the '{which}' property first.")

        return spline(soc)

    def get_dqdv(self, which: str, soc: npt.ArrayLike) -> npt.ArrayLike:
        """
        Evaluate a dqdv spline.

        Parameters
        ----------
        which : {'neg', 'pos', 'cell'}
            Which dqdv spline to evaluate.
        soc : ArrayLike
            Relative state-of-charge values, between [0, 1], to evaluate at.
            See notes for more information.

        Returns
        -------
        dvdq : np.ndarray
            Evaluated dqdv values.

        Raises
        ------
        ValueError
            'which' must be in ['neg', 'pos', 'cell'].
        RuntimeError
            If the requested spline has not yet been constructed.

        Notes
        -----
        The individual electrode datasets and splines are internally stored in
        opposite directions of one another. See the `get_ocv` method for more
        information to ensure you are evaluating each in the correct directions.

        """
        return 1 / self.get_dvdq(which, soc)

    def err_terms(self, params: npt.ArrayLike) -> RichResult:
        """
        Calculate errors between the fit and data.

        Parameters
        ----------
        params : ArrayLike, shape(n,)
            Array for xn0, xn1, xp0, xp1, and iR (optional). If you already
            performed a fit you can simply use `fit_result.x`.

        Returns
        -------
        errs : RichResult
            Voltage, dqdv, and dvdq errors. soc, fit, and data arrays are also
            included for convenience and plotting.

        Notes
        -----
        Errors are calculated as mean absolute percent errors between the data
        and fits. The normalization reduces preferences to fit any one cost
        term over others when more than one is considered. It also removes any
        units so it is more mathematically correct to sum the errors.

        """
        from ampworks.utils import RichResult

        self._check_initialized('err_terms')

        params = np.asarray(params)
        params[:4] = np.clip(params[:4], 0., 1.)

        if params.size == 5:
            xn0, xn1, xp0, xp1, iR = params
        else:
            xn0, xn1, xp0, xp1, iR = *params, 0.

        x_neg = xn0 + (xn1 - xn0) * self._soc
        x_pos = xp0 + (xp1 - xp0) * self._soc

        dxp_dx = xp1 - xp0  # for chain rule w.r.t. x_pos -> soc below
        dxn_dx = xn1 - xn0  # for chain rule w.r.t. x_neg -> soc below

        volt_fit = self._ocv_p(x_pos) - self._ocv_n(x_neg) - iR

        dvdq_fit = self._dvdq_p(x_pos)*dxp_dx - self._dvdq_n(x_neg)*dxn_dx
        dqdv_fit = 1 / dvdq_fit

        volt_data = self._volt_data
        dqdv_data = self._dqdv_data
        dvdq_data = self._dvdq_data

        volt_err = np.mean(np.abs((volt_fit - volt_data) / volt_data))
        dqdv_err = np.mean(np.abs((dqdv_fit - dqdv_data) / dqdv_data))
        dvdq_err = np.mean(np.abs((dvdq_fit - dvdq_data) / dvdq_data))

        # attempt at using relative MSE, non-trivial to figure out scaling...
        # volt_scale = np.maximum(volt_data, volt_data.mean())
        # dqdv_scale = np.maximum(dqdv_data, dqdv_data.mean())
        # dvdq_scale = np.maximum(dvdq_data, dvdq_data.mean())

        # volt_err = np.mean(((volt_fit - volt_data) / volt_scale)**2)
        # dqdv_err = np.mean(((dqdv_fit - dqdv_data) / dqdv_scale)**2)
        # dvdq_err = np.mean(((dvdq_fit - dvdq_data) / dvdq_scale)**2)

        errs = RichResult(
            soc=self._soc,
            volt_err=volt_err*100.,
            volt_fit=volt_fit,
            volt_data=volt_data,
            dqdv_err=dqdv_err*100.,
            dqdv_fit=dqdv_fit,
            dqdv_data=dqdv_data,
            dvdq_err=dvdq_err*100.,
            dvdq_fit=dvdq_fit,
            dvdq_data=dvdq_data,
        )

        return errs

    def grid_search(self, Nx: int) -> dict:
        """
        Determine the minimum error by evaluating parameter sets taken from
        intersections of a coarse grid. Parameter sets where either x0 < x1
        are ignored.

        Parameters
        ----------
        Nx : int
            Number of discretizations between [0, 1] for each parameter.

        Returns
        -------
        fit_result : :class:`~ampworks.dqdv.DqdvFitResult`
            Summarized results from the grid search.

        """
        from ampworks.dqdv import DqdvFitResult
        from ampworks.mathutils import combinations

        self._check_initialized('grid_search')

        span = np.linspace(0., 1., Nx)
        names = ['xn0', 'xn1', 'xp0', 'xp1', 'iR']

        params = combinations([span] * 4, names=names)

        valid_ps = []
        for p in params:
            if p['xn0'] < p['xn1'] and p['xp0'] < p['xp1']:
                valid_ps.append(p)

        errs = np.zeros(len(valid_ps))
        for i, p in enumerate(valid_ps):
            values = np.fromiter(p.values(), dtype=float)
            errs[i] = self._err_func(values)

        index = np.argmin(errs)
        x_opt = np.fromiter(valid_ps[index].values(), dtype=float)

        fit_result = DqdvFitResult(
            success=True,
            message='Done searching.',
            nfev=len(errs),
            niter=None,
            fun=errs[index],
            Ah=self.cell.Ah.max(),
            x=np.hstack([x_opt, 0.]),
            x_std=np.repeat(np.nan, 5),
            x_map=names,
        )

        return fit_result

    def constrained_fit(
        self, x0: npt.ArrayLike, bounds: float | list[float] = 0.1,
        xtol: float = 1e-8, maxiter: int = 100000, return_full: bool = False,
    ) -> DqdvFitResult:
        """
        Run a trust-constrained local optimization routine to minimize error
        between the fit and data.

        Parameters
        ----------
        x0 : ArrayLike, shape(n,)
            Initial xn0, xn1, xp0, xp1, and optionally iR. If you already ran
            a previous fit you can simply use `fit_result.x`.
        bounds : float or list[float], optional
            Symmetric parameter bounds (excludes iR). A float (default=0.1)
            applies to all. Use lists for per-x values. See notes for more info.
        xtol : float, optional
            Convergence tolerance for parameters. Defaults to 1e-8.
        maxiter : int, optional
            Maximum number of iteraterations. Defaults to 1e5.
        return_full : bool, optional
            If True, include the complete `OptimizeResult` from SciPy in the
            output. Defaults to False.

        Returns
        -------
        fit_result : :class:`~ampworks.dqdv.DqdvFitResult`
            A subset summary of SciPy's optimization results, including an added
            approximate standard deviation for the pameters.
        opt_result : OptimizeResult
            Full result form SciPy. Does not include standard deviation info.
            Only returned if `return_full=True`.

        Notes
        -----
        Bound indices correspond to xn0, xn1, xp0, and xp1, where 0 and 1 are
        in reference to lower and upper stoichiometries of the negative (n)
        and positive (p) electrodes. Set `bounds[i] = 1` to disable bounds
        and use the full interval [0, 1] for x[i]. If an `x[i] +/- bounds[i]`
        exceeds [0, 1], the lower and/or upper bounds will be corrected to 0
        and/or 1, respectively. Furthermore, bounds are clipped to be between
        0.001 and 1 behind the scenes. It does not help to use values outside
        this range.

        The `fit_result` output contains uncertainty estimates for the fitted
        parameters. These are approximated from the numerical Hessian at the
        optimum. The method assumes the function is locally linear, the input
        errors are independent and small, and the fit is well-behaved. Notes
        on the method are available `here <std-notes_>`_. These bounds provide
        more of a heuristic interpretation of the confidence intervals rather
        than a statistical interpretation. This is because all fitting routines
        use a mean absolute percent error (MAPE) function, but the uncertainty
        approximation needs a sum of squared residuals error function.

        Since two different error functions are used between the fit and the
        uncertainty estimates, they are not directly linked. However, using the
        combination of these two functions has empirically provided consistent
        convergence and reasonable uncertainties across a broad range of data
        sets. We highlight the details of the methods here and leave it to the
        user to interpret and decide whether or not to belience and/or use the
        uncertainty estimates.

        .. _std-notes:
            https://kitchingroup.cheme.cmu.edu/pycse/book/
            12-nonlinear-regression-2.html

        """
        from numdifftools import Hessian
        from ampworks.dqdv import DqdvFitResult

        self._check_initialized('constrained_fit')

        x0 = np.asarray(x0)
        eps = np.finfo(x0.dtype).eps

        # check and build bounds
        if isinstance(bounds, Real):
            bounds = [bounds]*4

        if not isinstance(bounds, Iterable):
            raise TypeError("'bounds' must be an iterable.")

        if len(bounds) != 4:
            raise ValueError("'bounds' must have length 4.")

        errs = self.err_terms(x0)

        iR0 = (errs['volt_fit'] - errs['volt_data']).mean()

        if x0.size == 5:
            x0[-1] = iR0
        elif x0.size == 4:
            x0 = np.hstack([x0, iR0])

        lower = np.zeros_like(x0)
        upper = np.ones_like(x0)
        bounds = np.clip(bounds, 1e-3, 1.)
        for i in range(4):
            lower[i] = max(0., x0[i] - bounds[i])
            upper[i] = min(1., x0[i] + bounds[i])

        if 'voltage' in self.cost_terms:
            lower[-1] = -np.inf
            upper[-1] = np.inf
        else:
            lower[-1] = -eps
            upper[-1] = eps

        bounds = [(L, U) for L, U in zip(lower, upper)]

        # constrain each x0 < x1
        constr_neg = opt.LinearConstraint([[1, -1, 0, 0, 0]], -np.inf, 0.)
        constr_pos = opt.LinearConstraint([[0, 0, 1, -1, 0]], -np.inf, 0.)

        constraints = [constr_neg, constr_pos]

        options = {
            'xtol': xtol,
            'maxiter': maxiter,
        }

        warnings.filterwarnings('ignore', 'delta_grad == 0.0')

        opt_result = opt.minimize(self._err_func, x0, method='trust-constr',
                                  bounds=bounds, constraints=constraints,
                                  options=options)

        warnings.filterwarnings('default')

        # Use Hessian to approximate variance. Requires SSR error function.
        # An added diagonal scaling stabilizes inversion (avoids non-singular).

        def ssr(x: npt.ArrayLike) -> float:  # sum of squared residuals

            errs = self.err_terms(x)

            volt_err = np.sum((errs['volt_fit'] - errs['volt_data'])**2)
            dqdv_err = np.sum((errs['dqdv_fit'] - errs['dqdv_data'])**2)
            dvdq_err = np.sum((errs['dvdq_fit'] - errs['dvdq_data'])**2)

            ssr = 0.
            if 'voltage' in self.cost_terms:
                ssr += volt_err
            if 'dqdv' in self.cost_terms:
                ssr += dqdv_err
            if 'dvdq' in self.cost_terms:
                ssr += dvdq_err

            return ssr

        opt_result.hess = Hessian(ssr)(opt_result.x)

        evals, _ = np.linalg.eig(opt_result.hess)
        scale = 1e-16*np.max(np.abs(evals))

        try:
            size = opt_result.x.size
            cov = np.linalg.inv(opt_result.hess + scale*np.eye(size))
            std = np.sqrt(0.5*np.abs(np.diag(cov)))
        except Exception:
            std = None

        if 'voltage' not in self.cost_terms:
            opt_result.x[-1] = 0.

            if std is not None:
                std[-1] = 0.

        fit_result = DqdvFitResult(
            success=opt_result.success,
            message=opt_result.message,
            nfev=opt_result.nfev,
            niter=opt_result.niter,
            fun=opt_result.fun,
            Ah=self.cell.Ah.max(),
            x=opt_result.x,
            x_std=std,
            x_map=['xn0', 'xn1', 'xp0', 'xp1', 'iR'],
        )

        if return_full:
            return fit_result, opt_result

        return fit_result

    def plot(self, params: npt.ArrayLike) -> None:
        """
        Plot the model fit vs. data.

        Parameters
        ----------
        params : ArrayLike, shape(n,)
            Array for xn0, xn1, xp0, xp1, and iR (optional). If you already
            performed a fit you can simply use `fit_result.x`.

        """
        from ampworks.utils import _ExitHandler
        from ampworks.plotutils import add_text, format_ticks, focused_limits

        self._check_initialized('plot')

        xn0, xn1, xp0, xp1 = params[:4]
        errs = self.err_terms(params)

        fig = plt.figure(figsize=[9.0, 3.75], constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        pstyle = {'ls': '-', 'lw': 2, 'color': 'C3', 'label': 'Pos'}
        nstyle = {'ls': '-', 'lw': 2, 'color': 'C0', 'label': 'Neg'}
        mstyle = {'ls': '-', 'lw': 2, 'color': 'k', 'label': 'Model'}
        dstyle = {'ls': '', 'ms': 7, 'marker': 'o', 'mfc': 'grey',
                  'alpha': 0.3, 'markeredgecolor': 'k', 'label': 'Data'}

        lines = []

        # ax1: pos, neg, model, and data voltages -----------------------------
        data = ax1.plot(errs['soc'][::5], errs['volt_data'][::5], **dstyle)
        model = ax1.plot(errs['soc'], errs['volt_fit'], **mstyle)

        lines.extend(data)
        lines.extend(model)

        socp = (errs['soc'] - xp0) / (xp1 - xp0)
        pos = ax1.plot(socp, self._ocv_p(errs['soc']), **pstyle)

        lines.extend(pos)

        twin = ax1.twinx()
        socn = (errs['soc'] - xn0) / (xn1 - xn0)
        neg = twin.plot(socn, self._ocv_n(errs['soc']), **nstyle)

        lines.extend(neg)

        # Vertical lines
        ax1.axvline(0., linestyle='--', color='grey')
        ax1.axvline(1., linestyle='--', color='grey')

        add_text(ax1, 0.5, 0.06, f"MAPE={errs['volt_err']:.2e}%", ha='center')

        ax1.set_xlabel(r"q [$-$]")
        ax1.set_ylabel(r"Voltage (pos/cell) [V]")
        twin.set_ylabel(r"Voltage (neg) [V]")

        ax1.legend(lines, [line.get_label() for line in lines], ncols=2,
                   loc='upper center', frameon=False)

        # ax2: dqdv -----------------------------------------------------------
        ax2.plot(errs['soc'], errs['dqdv_fit'], zorder=10, **mstyle)
        ax2.plot(errs['soc'][::3], errs['dqdv_data'][::3], **dstyle)

        add_text(ax2, 0.6, 0.85, f"MAPE={errs['dqdv_err']:.2e}%")

        ax2.set_xticklabels([])
        ax2.set_ylabel(r"dq/dV [1/V]")

        # ax3: dvdq -----------------------------------------------------------
        ax3.plot(errs['soc'], errs['dvdq_fit'], zorder=10, **mstyle)
        ax3.plot(errs['soc'][::3], errs['dvdq_data'][::3], **dstyle)

        add_text(ax3, 0.6, 0.85, f"MAPE={errs['dvdq_err']:.2e}%")

        ax3.set_xlabel(r"q [$-$]")
        ax3.set_ylabel(r"dV/dq [V]")

        dvdq = np.hstack([errs['dvdq_data'], errs['dvdq_fit']])
        ylims = focused_limits(dvdq)
        ax3.set_ylim(ylims)

        # additional formatting
        format_ticks(ax1, xdiv=2, ydiv=2, right=False)  # separate b/c twinx

        for ax in [twin, ax2, ax3]:
            format_ticks(ax, xdiv=2, ydiv=2)

        _ExitHandler.register_atexit(plt.show)
