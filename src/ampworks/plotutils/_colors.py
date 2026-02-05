from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import matplotlib as mpl

if TYPE_CHECKING:  # pragma: no cover
    from numpy import ndarray


class ColorMap:

    def __init__(self, cmap: str = 'jet', norm: tuple[float] = (0, 1)) -> None:
        """
        Map values to RGBA colors.

        A wrapper around matplotlib's colormaps to provide convenient access to
        generate normalized color sequences.

        Parameters
        ----------
        cmap : str, optional
            Name of the matplotlib colormap, by default 'jet'.
        norm : tuple[float], optional
            2-tuple of (min, max) values used for normalization.

        Raises
        ------
        ValueError
            'norm' must be length 2.
        ValueError
            'norm[0]' must be strictly less than 'norm[1]'.

        """
        if len(norm) != 2:
            raise ValueError("'norm' must be length 2.")
        elif norm[0] >= norm[1]:
            raise ValueError("'norm[0]' must be strictly less than 'norm[1]'.")

        cmap = mpl.colormaps[cmap]

        self._vmin, self._vmax = norm
        norm = mpl.colors.Normalize(vmin=self._vmin, vmax=self._vmax)

        self._sm = mpl.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)

    def get_color(self, scalar: float) -> tuple:
        """
        Map a scalar value to an RGBA color tuple.

        Parameters
        ----------
        scalar : float
            Value to map. Must lie within the normalization range.

        Returns
        -------
        color : tuple[float]
            The corresponding RGBA color.

        Raises
        ------
        ValueError
            'scalar' is not in bounds set by 'norm'.

        """
        if self._vmin <= scalar <= self._vmax:
            return self._sm.to_rgba(scalar)
        else:
            raise ValueError("'scalar' is not in bounds set by 'norm'.")


def colors_from_size(size: int, cmap: str = 'jet') -> list:
    """
    Generate a list of evenly spaced colors from a colormap.

    Parameters
    ----------
    size : int
        Number of colors to generate.
    cmap : str, optional
        Name of the matplotlib colormap, by default 'jet'.

    Returns
    -------
    colors : list
        List of RGBA color tuples.

    """
    colormap = ColorMap(cmap)
    data = np.linspace(colormap._vmin, colormap._vmax, size)

    return [colormap.get_color(i) for i in data]


def colors_from_data(data: ndarray, cmap: str = 'jet') -> ndarray:
    """
    Map an array of values to colors using a normalized colormap.

    Parameters
    ----------
    data : ndarray
        Input numeric array.
    cmap : str, optional
        Name of the matplotlib colormap, by default 'jet'.

    Returns
    -------
    colors : ndarray
        Array of RGBA color tuples with the same shape as 'data'.

    """
    colormap = ColorMap(cmap, norm=(data.min(), data.max()))
    colors = [colormap.get_color(x) for x in data.flatten()]

    return np.fromiter(colors, dtype=object).reshape(data.shape)
