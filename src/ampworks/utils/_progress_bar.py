from __future__ import annotations

from typing import Iterable

from tqdm import tqdm


class ProgressBar(tqdm):
    """Progress bar."""

    def __init__(self, iterable: Iterable = None, manual: bool = False,
                 desc: str = None, ncols: int = 80, total: int = None,
                 **kwargs) -> None:
        """
        Wraps the progress bar from `tqdm`, with different defaults. Also
        enables a custom "manual" mode in which the user manually sets the
        progress as a fraction in [0, 1] using `set_progress`.

        Parameters
        ----------
        iterable : Iterable, optional
            The iterable to use to construct the "automatic" progress bar, by
            default None. 'manual' must be False if 'iterable' is not None.
        manual : bool, optional
            True enables a "manual" mode progress bar, allowing manual updates
            via 'set_progress'. If False (default), 'iterable' cannot be None.
        desc : str, optional
            Prefix description, by default None.
        ncols : int, optional
            Terminal column width, by default 80. The special case of zero will
            display limited stats and time, with no progress bar.
        total : int, optional
            Number of expected iterations. Use when 'iterable' is a generator,
            otherwise estimated remaining time and the printed bar are skipped.
        **kwargs : dict, optional
            Additional keyword arguments to pass through to `tqdm`.

        Raises
        ------
        ValueError
            Provide exactly one of 'iterable' or 'manual', not both.

        Examples
        --------
        The examples below demonstrate the two uses of the `ProgressBar` class:
        the "automatic" mode with an iterable, and the "manual" mode with custom
        progress updates. As demonstrated, the `set_progress` method should be
        called once per "iteration" in the "manual" mode, and that the instance
        should be closed when finished.

        .. code-block:: python

            import time

            from ampworks.utils import ProgressBar

            # Automatic mode with an iterable
            for i in ProgressBar(range(5), desc='Iterable'):
                time.sleep(0.5)

            # Manual mode with custom progress updates
            progbar = ProgressBar(manual=True, desc='Manual')

            for i in range(5):
                time.sleep(0.5)
                progbar.set_progress((i + 1) / 5)

            progbar.close()

        """
        from ampworks._checks import _check_only_one

        _check_only_one(
            conditions=[manual, iterable is not None],
            message="Provide exactly one of 'iterable' or 'manual', not both.",
        )

        kwargs.setdefault('desc', desc)
        kwargs.setdefault('ncols', ncols)
        kwargs.setdefault('total', total)
        kwargs.setdefault('ascii', ' 2468█')
        kwargs.setdefault('iterable', iterable)

        self._iter = 0
        self._manual = manual
        if manual:
            kwargs['total'] = 1
            kwargs['bar_format'] = (
                "{l_bar}{bar}|{iter}[{elapsed}<{remaining}, {rate_fmt}]"
            )

        super().__init__(**kwargs)

    def set_progress(self, progress: float) -> None:
        """
        Updates the progress bar percentage and increments the tracked total
        number of iterations for the "manual" mode. Should be called once per
        "iteration", based on the user's definition of an iteration.

        Parameters
        ----------
        progress : float
            Progress fraction in [0, 1].

        """
        self._iter += 1
        self.n = progress
        self.refresh()

    def format_meter(self, n: int | float, total: int | float, elapsed: float,
                     **kwargs) -> str:
        """
        Wraps the parent `format_meter` method to customize stats for the
        "manual" mode. Users should not need to call this method directly.

        Parameters
        ----------
        n : int or float
            Number of finished iterations.
        total : int or float
            The expected total number of iterations. If meaningless (None),
            only basic progress statistics are displayed (no ETA).
        elapsed : float
            Number of seconds passed since start.
        **kwargs : dict, optional
            Extra keyword arguments to pass through to the parent method.

        Returns
        -------
        out : str
            Formatted meter and stats, ready to display.

        """
        if self._manual:
            kwargs['rate'] = self._iter / elapsed if elapsed > 0 else 0

            try:
                perc = n / total
                t = elapsed*(1 - perc) / perc

                m, s = divmod(int(t), 60)
                h, m = divmod(int(m), 60)

                w_hours = f"{h:d}:{m:02d}:{s:02d}"
                wo_hours = f"{m:02d}:{s:02d}"

                kwargs['remaining'] = w_hours if h else wo_hours

            except ZeroDivisionError:
                kwargs['remaining'] = '?'

        kwargs['iter'] = f" {self._iter}it "
        return super().format_meter(n, total, elapsed, **kwargs)

    def reset(self) -> None:
        """
        Resets the iteration count to zero for repeated use. Only works for
        manual mode. For iterables you will need to create a new instance.

        """
        self._iter = 0
        super().reset()

    def close(self) -> None:
        """
        Closes the progress bar and releases resources. Should be called when
        finished with the progress bar, especially in manual mode.

        """
        super().close()

    def __del__(self) -> None:  # stop warnings/errors from multiple close calls
        if hasattr(self, 'disable'):
            super().__del__()
