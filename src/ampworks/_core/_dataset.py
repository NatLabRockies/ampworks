from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px

from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource


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
        Downsample the dataset by eliminating rows given:

        - number of rows
        - fraction of rows
        - resolution of a specified column

        Parameters
        ----------
        n : int, optional
            Number of evenly spaced rows to keep, by default None.
        frac : float, optional
            Fraction (in (0, 1]) of evenly spaced rows to keep, by default None.
        resolution : tuple[str, float], optional
            Column (str) and resolution (float) to use for downsampling based on
            the absolute difference between adjacent values. By default None.
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
        The following demonstrates three ways to downsample a dataset. Note that
        only the `resolution` option requires a column to operate on.

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

    def interactive_plotly(
        self,
        x: str,
        y: str,
        *,
        tips: list[str] | None = None,
        figsize: tuple[int | None, int | None] = (800, 450),
        kind: Literal['line', 'scatter', 'both'] = 'line',
        save: str = None,
    ) -> None:
        """
        Create an interactive plotly figure with hover tips. Optionally save as
        a standalone HTML file, viewable without installing Python/ampworks.

        Parameters
        ----------
        x : str
            Column name for the variable to plot on the x-axis.
        y : str
            Column name for the variable to plot on the y-axis.
        tips : list[str] or None, optional
            List of column names to display as hover tips, by default None.
        figsize : tuple[int | None, int | None], optional
            Figure size (width, height) in pixels, by default (800, 450). Set
            either or both dimensions to None to allow them to stretch.
        kind : {'line', 'scatter', 'both'}, optional
            Kind of plot to draw. 'line' (default) for a line plot, 'scatter'
            for a scatter plot, or 'both' to show both a line and markers.
        save : str, optional
            File path to save the plot as an HTML file, by default None.

        See Also
        --------
        interactive_bokeh
            Interactive plots using bokeh. Typically has higher performance for
            large (100k - 1M) datasets and better support for notebook exports.

        Notes
        -----
        The responsive height size option is limited in notebook environments
        since output cells do not have adjustable heights. In these cases, the
        height is set to a default minimum value.

        Examples
        --------
        The following creates an interactive plot of an HPPC discharge dataset.
        Note that the x, y, and tips values must be existing columns; however,
        you can compute or add new columns before plotting, if needed, as shown
        by adding an 'Hours' column in the second figure below. Also, hovertips
        must be passed as a list, even if only one column is requested.

        .. code-block:: python

            import ampworks as amp

            data = amp.datasets.load_datasets('hppc/hppc_discharge')
            data.interactive_plotly('Seconds', 'Volts', tips=['Step'])

            # Add new column to plot time in hours instead of seconds
            data['Hours'] = data['Seconds'] / 3600
            data.interactive_plotly('Hours', 'Volts', tips=['Step', 'Amps'])

        """
        from ampworks.plotutils._plotly import (
            _apply_plotly_style, _render_plotly,
        )

        hover = {} if tips is None else {col: True for col in tips}

        kind = kind.lower()

        if kind in ['line', 'both']:
            markers = True if kind == 'both' else False
            fig = px.line(self, x=x, y=y, markers=markers, hover_data=hover)
        elif kind == 'scatter':
            fig = px.scatter(self, x=x, y=y, hover_data=hover)
        else:
            raise ValueError(
                "Invalid value for 'kind'. Expected one of {'line', 'scatter',"
                " 'both'}, but got " + f"{kind=}."
            )

        _apply_plotly_style(fig)
        _render_plotly(fig=fig, figsize=figsize, save=save)

    def interactive_bokeh(
        self,
        x: str,
        y: str,
        *,
        tips: list[str] | None = None,
        figsize: tuple[int | None, int | None] = (800, 450),
        kind: Literal['line', 'scatter', 'both'] = 'line',
        save: str = None,
    ) -> None:
        """
        Create an interactive bokeh figure with hover tips. Optionally save as
        a standalone HTML file, viewable without installing Python/ampworks.

        Parameters
        ----------
        x : str
            Column name for the variable to plot on the x-axis.
        y : str
            Column name for the variable to plot on the y-axis.
        tips : list[str] or None, optional
            List of column names to display as hover tips, by default None.
        figsize : tuple[int | None, int | None], optional
            Figure size (width, height) in pixels, by default (800, 450). Set
            either or both dimensions to None to allow them to stretch.
        kind : {'line', 'scatter', 'both'}, optional
            Type of plot to create. 'line' (default) for a line plot, 'scatter'
            for a scatter plot, or 'both' to show both a line and markers.
        save : str, optional
            File path to save the plot as an HTML file, by default None.

        See Also
        --------
        interactive_plotly
            Interactive plots using plotly. Typically has lower performance for
            large (100k - 1M) datasets, but is compatible with `dash` apps.

        Notes
        -----
        The responsive height size option is limited in notebook environments
        since output cells do not have adjustable heights. In these cases, the
        height is set to a default minimum value.

        Examples
        --------
        The following creates an interactive plot of an HPPC discharge dataset.
        Note that the x, y, and tips values must be existing columns; however,
        you can compute or add new columns before plotting, if needed, as shown
        by adding an 'Hours' column in the second figure below. Also, hovertips
        must be passed as a list, even if only one column is requested.

        .. code-block:: python

            import ampworks as amp

            data = amp.datasets.load_datasets('hppc/hppc_discharge')
            data.interactive_bokeh('Seconds', 'Volts', tips=['Step'])

            # Add new column to plot time in hours instead of seconds
            data['Hours'] = data['Seconds'] / 3600
            data.interactive_bokeh('Hours', 'Volts', tips=['Step', 'Amps'])

        """
        from ampworks.plotutils._bokeh import (
            BOKEH_CONFIG, _apply_bokeh_style, _render_bokeh,
        )

        if tips is None:
            tips = []

        kind = kind.lower()

        color = '#636EFA'  # adopt color from plotly's default

        cols = [x, y] + tips
        source = ColumnDataSource(data=self[cols])

        # Horizontal HTML tooltip to match Plotly's compact single-row layout
        tooltips = [(x, '$x'), (y, '$y')]
        for tip in tips:
            tooltips.append((tip, "@{" + tip + "}"))

        fig = figure(
            x_axis_label=x,
            y_axis_label=y,
            width=figsize[0],
            height=figsize[1],
            **BOKEH_CONFIG,
        )

        if kind not in ['line', 'scatter', 'both']:
            raise ValueError(
                "Invalid value for 'kind'. Expected one of {'line', 'scatter',"
                " 'both'}, but got " + f"{kind=}."
            )

        line = fig.line(x=x, y=y, source=source, color=color, line_width=2)

        # hide line if only scatter is requested, done for hover tool, to reduce
        # too many points showing when dense or overlapping (discussed below)
        if kind == 'scatter':
            line.glyph.line_alpha = 0

        if kind in ['scatter', 'both']:
            fig.scatter(x=x, y=y, source=source, color=color, size=4.5)

        # Attach hover only to the line so a single tooltip fires even when
        # markers are densely overlapping at zoomed-out views
        hover = HoverTool(mode='vline', renderers=[line], tooltips=tooltips)
        fig.add_tools(hover)

        _apply_bokeh_style(fig)
        _render_bokeh(fig=fig, figsize=figsize, save=save)

    def interactive_xy_plot(
        self,
        x: str,
        y: str,
        *,
        tips: list[str] | None = None,
        figsize: tuple[int | None, int | None] = (800, 450),
        save: str = None,
    ) -> None:
        """Deprecated. Use :meth:`interactive_plotly` instead."""
        import warnings
        warnings.warn(
            "interactive_xy_plot() is deprecated and will be removed in a "
            "future version. Use interactive_plotly() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.interactive_plotly(
            x=x, y=y, tips=tips, figsize=figsize, kind='both', save=save,
        )

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
            Values whose absolute value is below this threshold are set to zero.
            Note that values equal to the threshold are not zeroed.
        inplace : bool, optional
            Modify in place if True. If False (default), return a new Dataset.

        Returns
        -------
        data : Dataset or None
            The modified Dataset if 'inplace' is False. Otherwise, None.

        Examples
        --------
        Small non-zero values that can be attributed to noise can interfere with
        some analysis methods. For example, automatic pulse detection identifies
        pulses based on transitions from zero to non-zero current. This example
        filters out currents below 1% of the mean non-rest current, though the
        thresholds should be tailored to your specific use case and data.

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
        Shifts the `Seconds` column by subtracting the first row's value to set
        a new zero reference.

        Parameters
        ----------
        inplace : bool, optional
            Modify in place if True. If False (default), return a new Dataset.

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
        from ampworks._checks import _check_type, _check_columns

        _check_type('inplace', inplace, bool)
        _check_columns(self, ['Seconds'])

        result = self.copy()
        result['Seconds'] -= result['Seconds'].iloc[0]

        if inplace:
            self._update_inplace(result)
        else:
            return result
