from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes


def add_text(ax: Axes, xloc: float, yloc: float, text: str,
             ha: str = 'left', va: str = 'center') -> None:
    """
    Add text to ax at a given location.

    Parameters
    ----------
    ax : Axes
        An axis instance from a matplotlib figure.
    xloc : float
        Relative location (0-1) for text in x-direction.
    yloc : float
        Relative location (0-1) for text in y-direction.
    text : str
        Text string to add to figure.
    ha : str, optional
        Horizontal alignment relative to (xloc, yloc). Must be in {'left',
        'center', 'right'}. By default 'left'.
    va : str, optional
        Vertical alignment relative to (xloc, yloc). Must be in {'baseline',
        'bottom', 'center', 'center_baseline', 'top'}. By default 'center'.

    """
    ax.text(xloc, yloc, text, ha=ha, va=va, transform=ax.transAxes)
