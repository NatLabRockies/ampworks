"""
The `datasets` module provides example datasets bundled with `ampworks`. The
available functions allow users to list, download, or load in example datasets.
The datasets are used in tutorials and tests. They provide a convenient intro
to package functions without the overhead of requiring users to perform their
own experiments.

Most datasets were created using physics-based models like the pseudo-2D model.
A brief description of each dataset is given below:

1. `gitt_charge` - example GITT data (using charge/rest sequences)
2. `gitt_discharge` - example GITT data (using discharge/rest sequences)
3. `ici_charge` - example ICI data (using charge/rest sequences)
4. `ici_discharge` - example ICI data (using discharge/rest sequences)
5. `hppc_discharge` - example HPPC data (using discharge sequences)

"""

from __future__ import annotations

from typing import TYPE_CHECKING
from warnings import catch_warnings, filterwarnings

import os
import shutil
import pathlib

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import Dataset

__all__ = [
    'download_all',
    'list_datasets',
    'load_datasets',
]


def list_datasets() -> list[str]:
    """
    List names of available example datasets.

    Returns
    -------
    names : list[str]
        A list of example file names from an internal `resources` folder.

    """
    resources = pathlib.Path(os.path.dirname(__file__), 'resources')
    return os.listdir(resources)


def download_all(path: str | os.PathLike | None = None) -> None:
    """
    Copy example datasets into a local directory.

    Parameters
    ----------
    path : str or PathLike or None, optional
        Path to parent directory where a new `ampworks_datasets` folder will
        be created and example datasets will be copied to. If None (default),
        the current working directory is used.

    """
    resources = pathlib.Path(os.path.dirname(__file__), 'resources')

    path = pathlib.Path(path or '.').joinpath('ampworks_datasets')
    path.mkdir(parents=True, exist_ok=True)

    for name in list_datasets():
        orig = os.path.join(resources, name)
        new = os.path.join(path, name)

        shutil.copy(orig, new)


def load_datasets(*names: str) -> Dataset:
    """
    Load example datasets by name.

    Parameters
    ----------
    *names : str
        One or more dataset names to load.

    Returns
    -------
    datasets : Dataset or tuple[Dataset]
        A single dataset if one name, otherwise a tuple of datasets in the same
        order as the given `names`.

    Raises
    ------
    ValueError
        Requested dataset is not available.

    """
    from ampworks import read_csv

    available = list_datasets()
    resources = pathlib.Path(os.path.dirname(__file__), 'resources')

    datasets = []
    for name in names:

        if not name.endswith('.csv'):
            name += '.csv'

        if name not in available:
            raise ValueError(f"{name} is not an available dataset.")

        with catch_warnings():
            filterwarnings('ignore', message='.*No valid headers.*')
            data = read_csv(resources.joinpath(name))

        datasets.append(data)

    if len(datasets) == 1:
        return datasets[0]

    return tuple(datasets)
