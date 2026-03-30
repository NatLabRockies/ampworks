from __future__ import annotations

from warnings import warn
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px

if TYPE_CHECKING:  # pragma: no cover
    from typing import Sequence


class Dataset(pd.DataFrame):
    """General dataset."""

    @property
    def _constructor(self) -> Dataset:
        return Dataset

    @classmethod
    def from_csv(cls, filepath):
        from ampworks import read_csv
        return read_csv(filepath)

    @classmethod
    def from_excel(cls, filepath):
        from ampworks import read_excel
        return read_excel(filepath)

    @classmethod
    def from_table(cls, filepath):
        from ampworks import read_table
        return read_table(filepath)

    def downsample(
        self, *, n: int = None, frac: float = None,
        resolution: Sequence[str, float] = None,
        monotonic: bool = False, inplace: bool = False,
        ignore_index: bool = False, keep_last: bool = False,
    ) -> Dataset | None:
        """
        Downsample the dataset based on the specified criteria. Eliminates rows
        to reduce the dataset by one of the following methods:

        - Keep a given number of rows
        - Keep a given fraction of rows
        - Keep rows based on the resolution of a given column

        TODO: Implement monotonic option, see argument notes below.

        Parameters
        ----------
        n : int, optional
            Number of evenly spaced rows to keep, by default None.
        frac : float, optional
            Fraction of evenly spaced rows to keep, by default None.
        resolution : Sequence[str, float], optional
            Column (str) and resolution (float) to use for downsampling based on
            adjacent values. By default None.
        monotonic : bool, optional
            If True, enforces that the downsampled column results in a monotonic
            sequence. Default is False. NOT YET IMPLEMENTED...
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
            Specify exactly one of: n, frac, resolution.

        Examples
        --------
        >>> data = amp.datasets.load_datasets('dqdv/cell1_rough')
        >>>
        >>> # keep 100 evenly spaced rows
        >>> sample1 = data.downsample(n=100)
        >>>
        >>> # keep 50% of the data, dropping evenly spaced rows
        >>> sample2 = data.downsample(frac=0.5)
        >>>
        >>> # ensure adjacent voltage readings are at least 1 mV apart
        >>> sample3 = data.downsample(resolution=('Volts', 1e-3))

        """
        if sum(x is not None for x in [n, frac, resolution]) != 1:
            raise ValueError("Specify exactly one of: n, frac, resolution.")

        if monotonic:
            warn("'monotonic' option is not yet implemented.")

        df = self.copy()
        df = df.reset_index(drop=ignore_index)

        # keep a specified number of rows
        if n is not None:
            step = max(1, len(df) // n)
            mask = [i % step == 0 for i in range(len(df))]

        # keep a specified fraction of rows
        elif frac is not None:
            step = int(1 / frac)
            mask = [i % step == 0 for i in range(len(df))]

        # keep rows based on a resolution between adjacent values
        elif resolution is not None:
            column, atol = resolution

            mask = [True]  # always keep the first row
            last_val = df[column].iloc[0]
            for val in df[column].iloc[1:]:
                if np.abs(val - last_val) >= np.abs(atol):
                    mask.append(True)
                    last_val = val
                else:
                    mask.append(False)

        if keep_last:
            mask[-1] = True

        df = df[mask]

        if not ignore_index:
            df = df.set_index('index', drop=True)
            df.index.name = None

        if inplace:
            self.__init__(df)
        else:
            return df

    def interactive_xy_plot(
        self, x: str, y: str, tips: list[str] | None = None,
        figsize: Sequence[int, int] = (800, 450), save: str = None,
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
        figsize : Sequence[int, int], optional
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
        >>> data = amp.datasets.load_datasets('hppc/hppc_discharge')
        >>> data.interactive_xy_plot('Seconds', 'Volts', tips=['Step'])
        >>>
        >>> # Add new column to plot time in hours instead of seconds
        >>> data['Hours'] = data['Seconds'] / 3600
        >>> data.interactive_xy_plot('Hours', 'Volts', tips=['Step'])

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

    def zero_below(self, column: str, threshold: float,
                   inplace: bool = False) -> Dataset | None:
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
        >>> # zero out currents below a threshold from non-rest data
        >>> data = amp.datasets.load_datasets('hppc/hppc_discharge')
        >>> threshold = data.loc[data['State'] != 'R', 'Amps'].min()*1e-2
        >>> data_zeroed = data.zero_below(column='Amps', threshold=threshold)

        """
        df = self.copy()
        mask = df[column].abs() < abs(threshold)
        df.loc[mask, column] = 0.0

        if inplace:
            self.__init__(df)
        else:
            return df
