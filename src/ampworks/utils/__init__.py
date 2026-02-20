"""
The utilities module provides a collection of helper functions and classes
to simplify common tasks, such as timing specific portions of code, printing
progress bars in the console, and more.

"""

import atexit

from typing import Callable

from ._timer import Timer
from ._rich_table import RichTable
from ._rich_result import RichResult
from ._progress_bar import ProgressBar
from ._alphanum_sort import alphanum_sort

__all__ = [
    'Timer',
    'RichTable',
    'RichResult',
    'ProgressBar',
    'alphanum_sort',
]


class _ExitHandler:
    """
    Exit handler.

    Use this class to register functions that you want to run just before a
    file exits. This is primarily used to register plt.show() so plots appear
    in both interactive and non-interactive environments, even if the user
    forgets to explicitly call it.

    """

    _registered = []

    @classmethod
    def register_atexit(cls, func: Callable) -> None:
        if func not in cls._registered:
            cls._registered.append(func)
            atexit.register(func)
