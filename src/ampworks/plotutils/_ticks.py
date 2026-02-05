from __future__ import annotations
from typing import TYPE_CHECKING

from numpy import atleast_1d
from matplotlib.ticker import AutoMinorLocator

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes


def minor_ticks(ax: Axes, xdiv: int = None, ydiv: int = None) -> None:
    """
    Adds minor ticks.

    Parameters
    ----------
    ax : Axes
        An axis instance from a matplotlib figure.
    xdiv : int, optional
        Divisions between x major ticks. Defaults to None (auto locate).
    ydiv : int, optional
        Divisions between y major ticks. Defaults to None (auto locate).

    Notes
    -----
    This function ignores axes with logarithmic scaling.

    """
    axs = atleast_1d(ax)

    for ax in axs.flatten():
        if ax.get_xaxis().get_scale() != 'log':
            ax.xaxis.set_minor_locator(AutoMinorLocator(xdiv))

        if ax.get_yaxis().get_scale() != 'log':
            ax.yaxis.set_minor_locator(AutoMinorLocator(ydiv))


def tick_direction(ax: Axes, xdir: str = 'in', ydir: str = 'in',
                   top: bool = True, right: bool = True) -> None:
    """
    Controls tick directions.

    Parameters
    ----------
    ax : Axes
        An axis instance from a matplotlib figure.
    xdir : {'in', 'out', 'inout'}, optional
        Places x ticks inward, outward, or both. By default 'in'.
    ydir : {'in', 'out', 'inout'}, optional
        Places y ticks inward, outward, or both. By default 'in'.
    top : bool, optional
        Mirror the x ticks along the top of Axis, by default True.
    right : bool, optional
        Mirror the y ticks along the top of Axis, by default True.

    """
    axs = atleast_1d(ax)

    for ax in axs.flatten():
        ax.tick_params(axis='x', which='both', top=top, direction=xdir)
        ax.tick_params(axis='y', which='both', right=right, direction=ydir)


def format_ticks(
    ax: Axes, xdiv: int = None, ydiv: int = None, xdir: str = 'in',
    ydir: str = 'in', top: bool = True, right: bool = True,
) -> None:
    """
    Formats axis ticks.

    Specifically, applies both `minor_ticks()` and `tick_direction()` from
    one convenient function, instead of calling them separately.

    Parameters
    ----------
    ax : Axes
        An axis instance from a matplotlib figure.
    xdiv : int, optional
        Divisions between x major ticks. Defaults to None (auto locate).
    ydiv : int, optional
        Divisions between y major ticks. Defaults to None (auto locate).
    xdir : {'in', 'out', 'inout'}, optional
        Places x ticks inward, outward, or both. By default 'in'.
    ydir : {'in', 'out', 'inout'}, optional
        Places y ticks inward, outward, or both. By default 'in'.
    top : bool, optional
        Mirror the x ticks along the top of Axis, by default True.
    right : bool, optional
        Mirror the y ticks along the top of Axis, by default True.

    """
    minor_ticks(ax, xdiv=xdiv, ydiv=ydiv)
    tick_direction(ax, xdir=xdir, ydir=ydir, top=top, right=right)
