"""
The `datasets` module provides example datasets bundled with `ampworks`. The
available functions allow users to list, download, or load in example datasets.
The datasets are used in tutorials and tests. They provide a convenient intro
to package functions without the overhead of requiring users to perform their
own experiments.

Most datasets were created using physics-based models, like the pseudo-2D model.
Datasets are organized into subfolders by module, e.g., `ici` for ICI datasets.
A brief description of each dataset is given below:

GITT datasets:
    1. `gitt_charge` - example GITT data (using charge/rest sequences)
    2. `gitt_discharge` - example GITT data (using discharge/rest sequences)

HPPC datasets:
    1. `hppc_discharge` - example HPPC data (using discharge sequences)

ICI datasets:
    1. `ici_charge` - example ICI data (using charge/rest sequences)
    2. `ici_discharge` - example ICI data (using discharge/rest sequences)

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


def list_datasets(modules: str | list[str] | None = None) -> list[str]:
    """
    List names of available example datasets.

    Parameters
    ----------
    modules : str or list[str] or None, optional
        If given, only list datasets related to the given module(s) ('gitt',
        'ici', etc.). If None (default), list all available datasets.

    Returns
    -------
    names : list[str]
        A list of example file names from an internal `resources` folder.

    """
    resources = pathlib.Path(os.path.dirname(__file__), 'resources')
    subfolders = os.listdir(resources)

    if modules is None:
        modules = subfolders
    elif isinstance(modules, str):
        modules = [modules]

    missing = set(modules) - set(subfolders)
    if missing:
        raise ValueError(f"Requested module(s) not found: {missing}. Available"
                         f" modules are {subfolders}.")

    names = []
    for m in modules:
        files = [m + '/' + f for f in os.listdir(resources.joinpath(m))]
        names.extend(files)

    return names


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

    shutil.copytree(resources, path, dirs_exist_ok=True)


def load_datasets(*names: str) -> Dataset:
    """
    Load example datasets by name.

    Parameters
    ----------
    *names : str
        One or more dataset names to load. Check `list_datasets()` for available
        filenames. Note that including the '.csv' extension is optional.

    Returns
    -------
    datasets : Dataset or tuple[Dataset]
        A single dataset if one name, otherwise a tuple of datasets in the same
        order as the given `names`.

    Raises
    ------
    ValueError
        Requested dataset is not available.

    Examples
    --------
    >>> hppc_data = load_datasets('hppc/hppc_discharge.csv')
    >>> print(hppc_data)

    >>> ici_c, ici_d = load_datasets('ici/ici_charge', 'ici/ici_discharge')
    >>> print(ici_c)
    >>> print(ici_d)

    """
    from ampworks import read_csv

    available = list_datasets()
    resources = pathlib.Path(os.path.dirname(__file__), 'resources')

    names = [n + '.csv' if not n.endswith('.csv') else n for n in names]

    not_available = [n for n in names if n not in available]
    if not_available:
        raise ValueError(f"Requested dataset(s) not found: {not_available}.")

    datasets = []
    for name in names:
        with catch_warnings():
            filterwarnings('ignore', message='.*No valid headers.*')
            data = read_csv(resources.joinpath(name))

        datasets.append(data)

    if len(datasets) == 1:
        return datasets[0]

    return tuple(datasets)
