from __future__ import annotations

import time


class Timer:
    """Timer utility."""

    __slots__ = ('name', '_units', '_converter', '_start', '_stop',)

    def __init__(self, name: str = 'Elapsed time', units: str = 's') -> None:
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

    def __repr__(self) -> str:  # pragma: no cover

        data = {
            'name': self.name,
            'units': self._units,
            'start': self._start,
            'stop': self._stop,
            'elapsed': self.elapsed_time,
        }

        summary = "\n\t".join([f"{k}={v!r}," for k, v in data.items()])

        return f"Timer(\n{summary}\n)"

    def __enter__(self) -> Timer:
        """Store start time when entering "with" block."""
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Store stop time when exiting "with" block, and print."""
        self._stop = time.time()
        elapsed = self._converter[self._units](self.elapsed_time)

        print(f"{self.name}: {elapsed:.5f} {self._units}")

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
