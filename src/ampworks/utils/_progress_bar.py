from __future__ import annotations

from typing import Iterable

from tqdm import tqdm


class ProgressBar(tqdm):
    """Progress bar."""

    def __init__(
        self,
        iterable: Iterable | None = None,
        manual: bool = False,
        desc: str | None = None,
        ncols: int = 80,
        total: int | None = None,
        **kwargs,
    ) -> None:
        """
        Wraps `tqdm` with different defaults and enables a "manual" mode that is
        controlled using the `set_progress()` method.

        Parameters
        ----------
        iterable : Iterable or None, optional
            An iterable to automatically track progress across. Must be None if
            using the manual mode. The default is None. Set `total` when using a
            generator to make sure remaining time and the bar are displayed.
        manual : bool, optional
            True enables a manual mode where the user controls progress updates.
            Must be False (default) if `iterable` is not None.
        desc : str or None, optional
            Prefix description, by default None.
        ncols : int, optional
            Terminal column width, by default 80. The special case of zero will
            display limited stats and time, with no progress bar.
        total : int or None, optional
            Number of expected iterations. Use when `iterable` is a generator,
            otherwise estimated remaining time and the printed bar are skipped.
        **kwargs : dict, optional
            Additional keyword arguments to pass through to `tqdm`.

        Raises
        ------
        ValueError
            Provide exactly one of `iterable` or `manual`, not both.

        Examples
        --------
        The following examples demonstrate both the automatic and manual modes
        of the progress bar. Note that `set_progress()` must be called manually
        to update the progress using values between [0, 1] in manual mode.

        When assigning a progress bar to a variable, you should also call the
        `close()` method when finished using it to release resources. This is
        demonstrated for the manual mode below.

        .. code-block:: python

            import time

            from ampworks.utils import ProgressBar

            # automatic mode with an iterable
            for i in ProgressBar(range(5), desc='Iterable'):
                time.sleep(0.5)

            # manual mode with custom progress updates
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
        Updates progress in manual mode. Should be called once per iteration.

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
        Wraps the parent `format_meter` to customize stats in manual mode. Users
        should not need to call this method.

        Parameters
        ----------
        n : int or float
            Number of finished iterations.
        total : int or float
            The expected total number of iterations. If meaningless (None), only
            basic progress statistics are displayed (no ETA).
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
        Resets the iteration count to zero for repeated use. Only works for the
        manual mode. Create a new instance when using an iterable.

        """
        self._iter = 0
        super().reset()

    def close(self) -> None:
        """
        Closes the progress bar and releases resources. Should be called when
        finished with the progress bar, especially in manual mode.

        """
        super().close()

    def __del__(self) -> None:  # stop errors from multiple close calls
        if hasattr(self, 'disable'):
            super().__del__()
