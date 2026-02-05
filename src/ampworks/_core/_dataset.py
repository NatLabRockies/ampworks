from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px

if TYPE_CHECKING:  # pragma: no cover
    import numpy.typing as npt


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
        self, column: str, *, n: int = None, frac: int = None,
        resolution: float = None, inplace: bool = False,
        ignore_index: bool = False, keep_last: bool = False,
    ) -> Dataset:

        if sum(x is not None for x in [n, frac, resolution]) != 1:
            raise ValueError("Specify exactly one of: n, frac, resolution.")

        df = self.copy()
        df = df.reset_index(drop=ignore_index)

        if n is not None:
            step = max(1, len(df) // n)
            mask = [i % step == 0 for i in range(len(df))]

        elif frac is not None:
            step = int(1 / frac)
            mask = [i % step == 0 for i in range(len(df))]

        elif resolution is not None:
            mask = [True]  # always keep the first row
            last_val = df[column].iloc[0]
            for val in df[column].iloc[1:]:
                if np.abs(val - last_val) >= np.abs(resolution):
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
        figsize: npt.ArrayLike | None = (800, 450), save: str = None,
    ) -> None:

        from ampworks.plotutils._style import PLOTLY_TEMPLATE
        from ampworks.plotutils._render import _render_plotly

        if tips is None:
            tips = []

        fig = px.line(
            self, x=x, y=y, markers=True,
            hover_data={col: True for col in tips},
        )

        fig.update_layout(template=PLOTLY_TEMPLATE)
        _render_plotly(fig=fig, figsize=figsize, save=save)

    def zero_below(self, column: str, threshold: float,
                   inplace: bool = False) -> Dataset:
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
        data : Dataset
            The modified Dataset if 'inplace' is False.

        """
        df = self.copy()
        mask = df[column].abs() < abs(threshold)
        df.loc[mask, column] = 0.0

        if inplace:
            self.__init__(df)
        else:
            return df
