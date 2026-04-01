from __future__ import annotations

import time
import textwrap


class Timer:
    """Timer utility."""

    __slots__ = ('name', '_units', '_converter', '_start', '_stop', '_display')

    def __init__(
        self,
        name: str = 'Elapsed time',
        units: str = 's',
        display: bool = True,
    ) -> None:
        """
        Measures elapsed time for a series of steps and prints results to the
        console. Initialize with a name to tell what steps each print statement
        is associated with when timing multiple blocks. Also has control to
        print in different units of time.

        Parameters
        ----------
        name : str, optional
            Code block name used in print. The default is 'Elapsed time'.
        units : str, optional
            Printing units, from {'s', 'min', 'h'}. The default is 's'.
        display : bool, optional
            Whether to print the elapsed time when exiting a context block. The
            default is True.

        Notes
        -----
        If you want to print in multiple units, you can call `print_elapsed()`
        directly after exiting a context block. If your units are not seconds,
        minutes, or hours, you can also perform your own conversion using the
        `elapsed_time` property, which always returns seconds.

        A timer can be reused for multiple context blocks if desired, but the
        `elapsed_time` property will only return the most recent elapsed time
        because each time you enter a context block the start time is reset.
        So, make sure to print or store intermediate values if you want to keep
        track of multiple context blocks with a single timer.

        Examples
        --------
        The `Timer` works as a context manager and is accessed using `with`
        blocks. For example:

        .. code-block:: python

            import time

            from ampworks.utils import Timer

            def function(sleep_time: float) -> None:
                time.sleep(sleep_time)

            with Timer():
                function(2.)

        If you want to silence the print statement and just store the elapsed
        time, set `display=False` and access the `elapsed_time` property:

        .. code-block:: python

            with Timer(display=False) as timer:
                function(2.)

            print(f"Elapsed time: {timer.elapsed_time:.5f} s")

        """
        valid = ['s', 'min', 'h']
        if units not in valid:
            raise ValueError(f"{units=} is invalid; valid values are {valid}.")

        self.name = name
        self._units = units
        self._converter = {
            's': lambda t: t,
            'min': lambda t: t / 60.,
            'h': lambda t: t / 3600.,
        }

        self._start = 0.
        self._stop = 0.
        self._display = display

    def __repr__(self) -> str:  # pragma: no cover

        elapsed = self._converter[self._units](self.elapsed_time)
        elapsed_string = f"{elapsed} {self._units}"

        data = {
            'name': self.name,
            'elapsed': elapsed_string,
        }

        summary = "\n".join([f"{k}={v!r}," for k, v in data.items()])
        summary = textwrap.indent(summary, " " * 4)

        return f"{self.__class__.__name__}(\n{summary}\n)"

    def __enter__(self) -> Timer:
        """Store start time when entering "with" block."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Store stop time when exiting "with" block, and print."""
        self._stop = time.perf_counter()
        if self._display:
            self.print_elapsed(self._units)

    @property
    def elapsed_time(self) -> float:
        """
        Return the elapsed time in seconds.

        Returns
        -------
        elapsed : float
            Time difference between entering and exiting a 'with' block. Will
            return zero if it has not yet been used.

        """
        return self._stop - self._start

    def print_elapsed(self, units: str = 's') -> None:
        """
        Print the elapsed time.

        Parameters
        ----------
        units : str, optional
            Printing units, from {'s', 'min', 'h'}. The default is 's'.

        """
        elapsed = self._converter[units](self.elapsed_time)
        print(f"{self.name}: {elapsed:.5f} {units}")
