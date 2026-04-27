from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


class Dataset(pd.DataFrame):
    """General dataset."""

    @property
    def _constructor(self) -> Dataset:
        return Dataset

    def downsample(
        self,
        *,
        n: int = None,
        frac: float = None,
        resolution: tuple[str, float] = None,
        inplace: bool = False,
        ignore_index: bool = False,
        keep_last: bool = False,
    ) -> Dataset | None:
        """
        Downsample the dataset by eliminating rows using one of the following:

        - Keep a given number of rows
        - Keep a given fraction of rows
        - Keep rows based on the resolution of a given column

        Parameters
        ----------
        n : int, optional
            Number of evenly spaced rows to keep, by default None.
        frac : float, optional
            Fraction (in (0, 1]) of evenly spaced rows to keep, by default None.
        resolution : tuple[str, float], optional
            Column (str) and resolution (float) to use for downsampling based on
            adjacent values. By default None.
        inplace : bool, optional
            Modify in place if True. If False (default), return a new Dataset.
        ignore_index : bool, optional
            If True, reset the indices. Default is False.
        keep_last : bool, optional
            If True, always keep the last row. Default is False.

        Returns
        -------
        data : Dataset or None
            The downsampled Dataset if 'inplace' is False. Otherwise, None.

        Raises
        ------
        ValueError
            If more than one of n, frac, resolution is specified, or if they are
            all None. Also, if n is not positive or frac is not in (0, 1].

        Examples
        --------
        Below are examples of how to use the downsample method. In the first two
        examples, the rows are dropped evenly across the dataset. In the third
        example, rows are dropped based on the resolution of the 'Volts' column,
        ensuring that adjacent voltage readings are at least 1 mV apart.

        .. code-block:: python

            import ampworks as amp

            data = amp.datasets.load_datasets('dqdv/cell1_rough')

            # keep 100 evenly spaced rows
            sample1 = data.downsample(n=100)

            # keep 50% of the data, dropping evenly spaced rows
            sample2 = data.downsample(frac=0.5)

            # ensure adjacent voltage readings are at least 1 mV apart
            sample3 = data.downsample(resolution=('Volts', 1e-3))

        """
        from ampworks._checks import (
            _check_type, _check_only_one, _check_columns,
        )

        _check_only_one(
            conditions=[x is not None for x in [n, frac, resolution]],
            message="Specify exactly one of: n, frac, resolution.",
        )

        _check_type('inplace', inplace, bool)
        _check_type('ignore_index', ignore_index, bool)
        _check_type('keep_last', keep_last, bool)

        result = self.copy()

        if len(result) == 0:
            raise ValueError("Cannot downsample an empty dataset.")

        mask = np.zeros(len(result), dtype=bool)

        # keep a specified number of rows
        if n is not None:
            _check_type('n', n, int)

            if n <= 0:
                raise ValueError("'n' must be a positive integer.")

            count = min(n, len(result))
            indices = np.linspace(0, len(result) - 1, count, dtype=int)

        # keep a specified fraction of rows
        elif frac is not None:
            _check_type('frac', frac, (float, int))

            if not (0 < frac <= 1):
                raise ValueError("'frac' must be in the range (0, 1].")

            count = int(len(result) * frac) or 1  # keep at least one row
            indices = np.linspace(0, len(result) - 1, count, dtype=int)

        # keep rows based on a resolution between adjacent values
        elif resolution is not None:
            _check_type('resolution', resolution, (tuple, list))

            if len(resolution) != 2:
                raise ValueError("'resolution' must be length 2.")

            _check_type('resolution[0]', resolution[0], str)
            _check_type('resolution[1]', resolution[1], (float, int))

            column, atol = resolution
            _check_columns(result, [column])

            column_data = result[column].to_numpy()

            indices = [0]  # always keep the first row
            last_val = column_data[0]
            for i, val in enumerate(column_data[1:], start=1):
                if np.abs(val - last_val) >= np.abs(atol):
                    indices.append(i)
                    last_val = val

        mask[indices] = True

        if keep_last:
            mask[-1] = True

        result = result[mask]

        if ignore_index:
            result = result.reset_index(drop=True)

        if inplace:
            self._update_inplace(result)
        else:
            return result

    def enforce_monotonic(
        self,
        column: str,
        increasing: bool = True,
        strict: bool = False,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Dataset | None:
        """
        Enforce monotonicity in a column by dropping rows that break the trend.

        Parameters
        ----------
        column : str
            Column name to enforce monotonicity on.
        increasing : bool, optional
            If True (default), enforce increasing monotonicity. Otherwise, apply
            decreasing monotonicity.
        strict : bool, optional
            If True, enforce strict monotonicity (no equal adjacent values). The
            default is False, which allows equal adjacent values.
        inplace : bool, optional
            Modify in place if True. If False (default), return a new Dataset.
        ignore_index : bool, optional
            If True, reset the indices. Default is False.

        Returns
        -------
        data : Dataset or None
            The modified Dataset if 'inplace' is False. Otherwise, None.

        """
        from ampworks._checks import _check_type, _check_columns

        _check_type('column', column, str)
        _check_type('increasing', increasing, bool)
        _check_type('strict', strict, bool)
        _check_type('inplace', inplace, bool)
        _check_type('ignore_index', ignore_index, bool)

        result = self.copy()

        if len(result) == 0:
            raise ValueError("Cannot enforce monotonicity on an empty dataset.")

        # loop over indices and store which to keep
        mask = np.zeros(len(result), dtype=bool)

        if increasing:
            compare = np.greater if strict else np.greater_equal
        else:
            compare = np.less if strict else np.less_equal

        _check_columns(result, [column])

        column_data = result[column].to_numpy()

        indices = [0]  # always keep the first row
        last_val = column_data[0]
        for i, val in enumerate(column_data[1:], start=1):
            if compare(val, last_val):
                indices.append(i)
                last_val = val

        # keep the rows where the monotonicity condition is met
        mask[indices] = True

        result = result[mask]

        if ignore_index:
            result = result.reset_index(drop=True)

        if inplace:
            self._update_inplace(result)
        else:
            return result

    def interactive_xy_plot(
        self, x: str, y: str, tips: list[str] | None = None,
        figsize: tuple[int, int] = (800, 450), save: str = None,
    ) -> None:
        """
        Create an interactive XY plot using Plotly. Allows hovertips, zooming,
        and more. Optionally, save the plot to an html file, which can be sent
        and opened in a web browser, without needing Python and/or ampworks.

        The hovertips are particularly useful for exploring the data and finding
        specific cycle and steps for slicing and further analysis.

        Parameters
        ----------
        x : str
            Column name for the variable to plot on the x-axis.
        y : str
            Column name for the variable to plot on the y-axis.
        tips : list[str] or None, optional
            List of column names to display as hover tips, by default None.
        figsize : tuple[int, int], optional
            Figure size (width, height) in pixels, by default (800, 450).
        save : str, optional
            File path to save the plot as an HTML file, by default None.

        Notes
        -----
        When run inside a Jupyter notebook, the plot will be rendered inline. If
        instead this function is called from a script, the plot will be saved to
        a temporary directory and automatically opened in a local web browser.

        Examples
        --------
        The following example uses the 'hppc_discharge' dataset and creates an
        interactive XY plot of 'Seconds' vs. 'Volts', with a hover tip showing
        the step number. Even though only one hover tip is requested, it must
        be passed in a list. For more than one hover tip, simply add more column
        names to the list.

        The interactive plots only allow one x and one y variable, and both are
        required to be existing columns in the dataset. In the second example,
        we compute a new column for time in hours so that we can change the
        x-axis to 'Hours' instead of 'Seconds'.

        .. code-block:: python

            import ampworks as amp

            data = amp.datasets.load_datasets('hppc/hppc_discharge')
            data.interactive_xy_plot('Seconds', 'Volts', tips=['Step'])

            # Add new column to plot time in hours instead of seconds
            data['Hours'] = data['Seconds'] / 3600
            data.interactive_xy_plot('Hours', 'Volts', tips=['Step'])

        """
        from ampworks.plotutils._plotly import PLOTLY_TEMPLATE, _render_plotly

        if tips is None:
            tips = []

        fig = px.line(
            self, x=x, y=y, markers=True,
            hover_data={col: True for col in tips},
        )

        fig.update_layout(template=PLOTLY_TEMPLATE)
        _render_plotly(fig=fig, figsize=figsize, save=save)

    def zero_below(
        self,
        column: str,
        threshold: float,
        inplace: bool = False,
    ) -> Dataset | None:
        """
        Set values in 'column' below 'threshold' to zero.

        Parameters
        ----------
        column : str
            Column name to apply thresholding.
        threshold : float
            Values with absolute value below this threshold are set to zero.
        inplace : bool, optional
            If True, modify the Dataset in place. Otherwise, return a new
            Dataset. Default is False.

        Returns
        -------
        data : Dataset or None
            The modified Dataset if 'inplace' is False. Otherwise, None.

        Examples
        --------
        Occasionally, there may be small non-zero values in the data that can
        be considered as noise and set to zero. When not appropriately zeroed,
        these can cause issues with automatic pulse detection (i.e., where the
        algorithm detects changes from rests to non-rests and vice versa). So,
        in the following example, we load the 'hppc_discharge' dataset and zero
        out current values below a certain threshold. The threshold here is set
        to 1% of the mean current from non-rest data, however, the appropriate
        threshold should be determined based on the specific characteristics of
        the dataset.

        .. code-block:: python

            import ampworks as amp

            # zero out currents below a threshold from non-rest data
            data = amp.datasets.load_datasets('hppc/hppc_discharge')
            threshold = data.loc[data['State'] != 'R', 'Amps'].mean()*1e-2

            data_zeroed = data.zero_below(column='Amps', threshold=threshold)

        """
        result = self.copy()
        mask = result[column].abs() < abs(threshold)
        result.loc[mask, column] = 0.0

        if inplace:
            self._update_inplace(result)
        else:
            return result

    def zero_time(self, inplace: bool = False) -> Dataset | None:
        """
        Shift the `Seconds` column by subtracting the value in the first row,
        creating a new zero time reference.

        Parameters
        ----------
        inplace : bool, optional
            If True, modify the Dataset in place. Otherwise, return a new
            Dataset. Default is False.

        Returns
        -------
        data : Dataset | None
            The modified Dataset if 'inplace' is False. Otherwise, None.

        Notes
        -----
        This method does not sort by time, nor does it use the minimum time when
        subtracting. It simply shifts the time values so that the first row has
        a time of zero, regardless of the actual order of time values. Consider
        sorting by time first, if needed, using `data.sort_values('Seconds')`.

        """
        from ampworks._checks import _check_columns

        _check_columns(self, ['Seconds'])

        result = self.copy()
        result['Seconds'] -= result['Seconds'].iloc[0]

        if inplace:
            self._update_inplace(result)
        else:
            return result
