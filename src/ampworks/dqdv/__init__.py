"""
Functions for analyzing dQdV data from battery experiments. Provides curve
extraction and smoothing, fitting methods, and post-processing of degradation
modes from fitted stoichiometries.

"""

import warnings

from ._dqdv_fitter import DqdvFitter
from ._dqdv_spline import DqdvSpline
from ._tables import DqdvFitResult, DqdvFitTable, DegModeTable
from ._lam_lli import calc_lam_lli, plot_lam_lli

__all__ = [
    'DqdvFitter',
    'DqdvSpline',
    'DqdvFitResult',
    'DqdvFitTable',
    'DegModeTable',
    'calc_lam_lli',
    'plot_lam_lli',
    'run_gui',
]


def run_gui(jupyter_mode: str = 'external', jupyter_height: int = 650) -> None:
    """
    Run a graphical dQdV fitting interface.

    Parameters
    ----------
    jupyter_mode : {'external', 'inline'}, optional
        How to display the GUI in jupyter notebooks. Run in a new browser tab
        with 'external' (default), or in the notebook with 'inline'.
    jupyter_height : int, optional
        Height (in px) when displayed using 'inline'. Defaults to 650.

    Warnings
    --------
    This function is only intended for use inside Jupyter Notebooks. You may
    experience issues if you call it from a normal script, or in an interactive
    session within some IDEs (e.g., Spyder, PyCharm, IPython, etc.). if you're
    looking for another way to access the GUI without needing to open Jupyter
    Notebooks, you can use the `ampworks --app` command from your terminal.

    """
    from ampworks.dqdv.gui_files import _gui
    from ampworks import _in_interactive, _in_notebook

    if not isinstance(jupyter_mode, str):
        raise TypeError("'jupyter_mode' must be type str.")
    elif jupyter_mode not in ['external', 'inline']:
        raise ValueError("'jupyter_mode' must be in {'external', 'inline'}.")

    if not isinstance(jupyter_height, int):
        raise TypeError("'jupyter_height' must be type int.")

    if not _in_notebook():
        jupyter_mode = 'external'

    if _in_interactive() and not _in_notebook():
        warnings.warn(
            "It looks like you're calling `run_gui()` from an interactive"
            " environment (e.g., Spyder, IPython, etc.). If the GUI fails,"
            " try calling the function inside a Jupyter Notebook instead."
            " Or, use the `ampworks --app` command in your terminal."
        )

    _gui.run(jupyter_mode=jupyter_mode, jupyter_height=jupyter_height)
