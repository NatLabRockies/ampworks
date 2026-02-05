from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike


def focused_limits(
    y: ArrayLike, factor: float = 2.5, margin: float = 0.05,
) -> tuple[float, float]:
    """
    Compute limits by ignoring outliers.

    This function determines the interquartile range (IQR) of data and uses
    that to determine outliers. Suggested limits are then provided based on
    data that falls within some factor of the IQR. While the first input is
    named 'y', this function is equally valid for suggesting 'x' limits.

    Parameters
    ----------
    y : ArrayLike
        Input data values.
    factor : float, optional
        Multiplier for the IQR. Defaults to 2.5. Any values outside the range
        `[Q1 - factor*IQR, Q3 + factor*IQR]` are considered outliers.
    margin : float, optional
        Fractional padding to add to the final limits, relative to the data
        range after clipping. Default is 0.05 (5%).

    Returns
    -------
    ymin, ymax : float
        Suggested axis limits for plotting.

    """
    y = np.asarray(y)
    Q1, Q3 = np.percentile(y, [25, 75])
    IQR = Q3 - Q1
    mask = (y >= Q1 - factor * IQR) & (y <= Q3 + factor * IQR)
    y_clip = y[mask]

    if y_clip.size == 0:  # fallback if everything is clipped
        y_clip = y

    ymin, ymax = y_clip.min(), y_clip.max()
    pad = margin * (ymax - ymin)
    return ymin - pad, ymax + pad


# def shared_limits(axs, axis: str = 'y', margin: float = 0.0) -> None:
#     # Compute shared min/max across multiple axes and set the same limits,
#     # optionally adding a margin fraction for padding.
#     ...
