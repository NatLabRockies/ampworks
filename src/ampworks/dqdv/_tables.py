from __future__ import annotations

import pandas as pd

from ampworks.utils import RichTable, RichResult


class DqdvFitResult(RichResult):
    """Container for a single dQdV fit."""

    def __init__(self, **kwargs) -> None:
        """
        Container for fits from the `DqdvFitter`. An instance of this class
        is returned from both the grid search and constrained fit methods.

        ========= ========= ==============================================
        Attribute Type      Description
        ========= ========= ==============================================
        success   bool      whether or not the routine exited successfully
        message   str       description of the cause of termination
        nfev      int       number of function evaluations
        niter     int       number of optimization iterations
        fun       float     value of objective function at `x`
        Ah        float     max capacity (Ah) of the fitted `cell` dataset
        x         1D array  the solution of the optimization
        x_std     1D array  approximate standard deviation for each `x`
        x_map     list[str] names/order of values store in `x`
        ========= ========= ==============================================

        """
        super().__init__(**kwargs)


class DqdvFitTable(RichTable):
    """Container for many dQdV fits."""

    _required_cols = [
        'Ah', 'xn0', 'xn0_std', 'xn1', 'xn1_std', 'xp0', 'xp0_std',
        'xp1', 'xp1_std', 'iR', 'iR_std', 'fun', 'success', 'message',
    ]

    def __init__(self, extra_cols: list[str] | None = None) -> None:
        """
        A container to store multiple dQdV fits. An instance of this class is
        required to run `calc_lam_lli`. Loop over multiple datasets and append
        fits one at a time to the table using the `append` method.

        Parameters
        ----------
        extra_cols : list[str] or None, optional
            Any extra, non-required columns to add to the table. Pass the column
            names and their row values to `append` when writing each row. Use
            to track equivalent full cycles or other metrics with each fit.

        Raises
        ------
        TypeError
            'extra_cols' must be type list[str].

        """
        if extra_cols is None:
            extra_cols = []

        if not isinstance(extra_cols, list):
            raise TypeError("'extra_cols' must be type list[str].")

        self._extra_cols = extra_cols
        data = {col: [] for col in self._required_cols + self._extra_cols}

        df = pd.DataFrame(data)

        super().__init__(df)

    def append(self, fit_result: DqdvFitResult, **extra_cols) -> None:
        """
        Append a new row to the table.

        Parameters
        ----------
        fit_result : DqdvFitResult
            A result from the `DqdvFitter's` grid search or constrained fit.
        extra_cols : dict, optional
            Any extra column names/values to include in the row.

        Raises
        ------
        ValueError
            Columns cannot be created on the fly. Any extra columns must be
            defined at initialization.

        See Also
        --------
        ~ampworks.dqdv.DqdvFitResult : Container for a single dQdV fit.

        """
        row = {
            'Ah': fit_result.Ah,
            'fun': fit_result.fun,
            'success': fit_result.success,
            'message': fit_result.message,
        }

        # fill from x_map
        for idx, name in enumerate(fit_result.x_map):
            row[name] = fit_result.x[idx]
            row[name + '_std'] = fit_result.x_std[idx]

        # add in any extra columns
        for k in extra_cols.keys():
            if k not in self.df.columns:
                raise ValueError(
                    f"Column '{k}' does not exist in 'DqdvFitResult'. Extra"
                    " columns must be defined during initialization."
                )

            row[k] = extra_cols[k]

        # append the new row
        self.df.loc[len(self.df)] = row


class DegModeTable(RichTable):
    """Degradation modes table."""

    _required_cols = [
        'Qn', 'Qn_std', 'Qp', 'Qp_std', 'Qc', 'LAMn', 'LAMn_std',
        'LAMp', 'LAMp_std', 'LLI', 'LLI_std',
    ]

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Output container for `calc_lam_lli`. Stores capacities (Ah), loss of
        active material (LAM), loss of lithium inventory (LLI), and standard
        deviations (std). 'n', 'p', and 'c' in the column names refer to the
        negative electrode, positive electrode, and full cell, respectively.
        May also include extra columns inherited from the `DqdvFitTable`, if
        present.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe to store.

        See Also
        --------
        ~ampworks.dqdv.calc_lam_lli : Calculate degradation modes from fits.

        """
        super().__init__(df)
