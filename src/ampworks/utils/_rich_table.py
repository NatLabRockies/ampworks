from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path
    from typing import Any, Self, Sequence
    from pandas import Series, DataFrame


class RichTable:
    """DataFrame-like results container."""

    _required_cols: Sequence[str] = []  # override in subclasses

    def __init__(self, df: DataFrame) -> None:
        """
        Provides a structured way to store data using a `pd.DataFrame` with
        additional validation and formatting features. Use this class directly
        by passing in a `pd.DataFrame`, or subclass it to define custom
        containers with required columns.

        Inheriting classes should define the class attribute `_required_cols`
        which is a list of column names that must be present in the input. If
        any are missing, initialization will raise a `ValueError`.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe to store. Columns are validated against the
            `_required_cols` attribute and then stored.

        Notes
        -----
        While this container is meant to act as a simplified dataframe, access
        to the full `pd.DataFrame` is provided via the `df` property. While
        the entire dataframe cannot be replaced, it can be manipulated in place
        through this property.

        Examples
        --------
        A minimal example using `RichTable` directly:

        .. code-block:: python

            import pandas as pd
            from ampworks.utils import RichTable

            df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
            table = RichTable(df)
            print(table)

        Subclassing to enforce required columns:

        .. code-block:: python

            import pandas as pd
            from ampworks.utils import RichTable

            class CustomTable(RichTable):
                _required_cols = ['Seconds', 'Volts']

            df = pd.DataFrame({'Seconds': [0, 1], 'Volts': [3.8, 3.7]})
            table = CustomTable(df)  # valid, no errors raised

        """
        self._validate_columns(df)
        self._df = df.copy()

    def __getitem__(self, key: str | list[str]) -> Series | DataFrame:
        """Retrieve one of more columns by key."""
        return self._df[key]

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError(
                f"Cannot set attribute {name}. Use 'df' to modify columns."
            )

    def __getattr__(self, name: str) -> Series:
        """Provide attribute-style access to columns."""
        df = object.__getattribute__(self, '_df')
        if name in df.columns:
            return df[name]
        raise AttributeError(name)

    def __repr__(self) -> str:
        """Return the string representation of the underlying DataFrame."""
        return repr(self._df)

    @classmethod
    def _validate_columns(cls, df: DataFrame) -> None:
        """
        Ensure that all required columns are present.

        Parameters
        ----------
        df : pd.DataFrame
            The input DataFrame to validate during initialization.

        Raises
        ------
        ValueError
            If any columns listed in `_required_cols` are missing.

        """
        required = set(cls._required_cols)
        if not required.issubset(df.columns):
            raise ValueError(f"Missing columns, {required=}.")

    @classmethod
    def from_csv(cls, path: str | Path) -> Self:
        """
        Create a new instance from a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to the CSV file.

        Returns
        -------
        table : Self
            A new instance initialized with data from the file.

        """
        from pandas import read_csv
        df = read_csv(path)

        return cls(df)

    @property
    def df(self) -> DataFrame:
        """The underlying DataFrame stored in the container."""
        return self._df

    def to_csv(self, path: str | Path) -> None:
        """
        Write the table to a CSV file.

        Parameters
        ----------
        path : str or Path
            Path to the output file.

        """
        self._df.to_csv(path, index=False)

    def copy(self) -> Self:
        """
        Returns a copy of the instance.

        Returns
        -------
        table : Self
            A deep copy of the instance. Does not share memory with original.

        """
        from copy import deepcopy
        return deepcopy(self)
