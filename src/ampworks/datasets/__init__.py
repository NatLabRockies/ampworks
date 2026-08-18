"""
The `datasets` module provides example datasets bundled with `ampworks`. The
available functions allow users to list, download, or load in example datasets.
The datasets are used in tutorials and tests. They provide a convenient intro
to package functions without the overhead of requiring users to perform their
own experiments.

Datasets come from combinations of real-world experiments and model-generated
data (ECM, SPM, P2D). While the `ampworks` algorithms are designed to work with
real experimental data, model-generated data has also been useful in testing and
demonstrating the algorithms in a controlled setting. Note that the included
datasets are not intended to cover all user cases, and users are encouraged to
apply the algorithms to their own data after learning from examples. Datasets
are organized into subfolders by module, e.g., `ici` for ICI datasets. A brief
description of each dataset is given below:

dQdV datasets:
    1. `cell1_rough` - noisy beginning of life full cell pseudo-OCV curve
    2. `cell1_smooth` - smoothed version of `cell1_rough`
    3. `cell2_rough` - noisy aged full cell pseudo-OCV curve
    4. `cell2_smooth` - smoothed version of `cell2_rough`
    5. `gr_smooth` - smoothed graphite electrode pseudo-OCP voltage curve
    6. `nmc_smooth` - smoothed NMC electrode pseudo-OCP voltage curve

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

from typing import TYPE_CHECKING, Literal

import os
import shutil
import pathlib

import pandas as pd

if TYPE_CHECKING:
    from ampworks import Dataset

__all__ = [
    'download_all',
    'list_datasets',
    'load_datasets',
]

RESOURCES = pathlib.Path(os.path.dirname(__file__), 'resources')
DATAFOLDERS = os.listdir(RESOURCES)


def list_datasets(*modules: str) -> list[str]:
    """
    List names of available example datasets.

    Parameters
    ----------
    modules : str, optional
        If given, only list datasets related to the given module(s) ('gitt',
        'ici', etc.). Leaving empty (default) lists all datasets.

    Returns
    -------
    names : list[str]
        A list of example file names from an internal `resources` folder.

    Raises
    ------
    ValueError
        Requested module(s) not found or empty. See the list of modules that
        have available datasets by printing `ampworks.datasets.DATAFOLDERS`.

    Examples
    --------
    The code snippets below show how to use the `list_datasets` function to list
    available datasets. The first example lists all datasets, while the second
    and third examples filter the list by module name.

    >>> names = list_datasets()
    >>> print(names)

    >>> names = list_datasets('gitt')
    >>> print(names)

    >>> names = list_datasets('gitt', 'ici')
    >>> print(names)

    """
    if not modules:
        modules = DATAFOLDERS

    missing = set(modules) - set(DATAFOLDERS)
    if missing:
        raise ValueError(f"Requested module(s) not found, or empty: {missing=}."
                         f" Available modules are {DATAFOLDERS=}.")

    names = []
    for m in modules:
        files = [m + '/' + f for f in os.listdir(RESOURCES.joinpath(m))]
        names.extend(files)

    return names


def download_all(
    path: str | os.PathLike | None = None,
    format: Literal['csv', 'parquet', 'txt', 'xlsx'] = 'parquet',
) -> None:
    """
    Copy example datasets into a local directory.

    Parameters
    ----------
    path : str or PathLike or None, optional
        Path to parent directory where a new `ampworks_datasets` folder will
        be created and example datasets will be copied to. If None (default),
        the current working directory is used.
    format : Literal['csv', 'parquet', 'txt', 'xlsx'], optional
        Format for the downloaded files, default is 'parquet'.

    """
    from ampworks._checks import _check_literal

    format = format.lower().lstrip()
    _check_literal('format', format, {'csv', 'parquet', 'txt', 'xlsx'})

    path = pathlib.Path(path or '.').joinpath('ampworks_datasets')
    path.mkdir(parents=True, exist_ok=True)

    # just copy if default parquet, return stops from executing further
    if format == 'parquet':
        shutil.copytree(RESOURCES, path, dirs_exist_ok=True)
        return

    # read and write to alternative format if not parquet
    writers = {
        'csv': lambda df, p: df.to_csv(p, index=False),
        'txt': lambda df, p: df.to_csv(p, sep='\t', index=False),
        'xlsx': lambda df, p: df.to_excel(p, index=False),
    }

    for file in pathlib.Path(RESOURCES).glob('*.parquet'):
        df = pd.read_parquet(file)
        writers[format](df, path / f"{file.stem}.{format}")


def load_datasets(*names: str) -> Dataset:
    """
    Load example datasets by name.

    Parameters
    ----------
    *names : str
        One or more dataset names to load. Check `list_datasets()` for available
        filenames. Note that including the '.parquet' extension is optional.

    Returns
    -------
    data : Dataset or tuple[Dataset]
        A single dataset if one name, otherwise a tuple of datasets in the same
        order as the given `names`.

    Raises
    ------
    ValueError
        Requested dataset is not available.

    Examples
    --------
    In the following example, the `load_datasets` function is used to load a
    single HPPC dataset and the optional `.parquet` extension is included. The
    names for available datasets can be found using `list_datasets`.

    >>> hppc_data = load_datasets('hppc/hppc_discharge.parquet')

    In the next example, two ICI datasets are loaded at once by providing their
    names. Here, the `.parquet` extensions is omitted, but the function appends
    it as needed. The returned datasets are provided in the same order as the
    given names.

    >>> ici_c, ici_d = load_datasets('ici/ici_charge', 'ici/ici_discharge')

    """
    from ampworks import Dataset

    available = list_datasets()

    if len(names) == 0:
        raise ValueError("At least one dataset name must be given.")

    names = [n + '.parquet' if not n.endswith('.parquet') else n for n in names]

    not_available = [n for n in names if n not in available]
    if not_available:
        raise ValueError(f"Requested dataset(s) not found: {not_available}.")

    datasets = []
    for name in names:
        df = pd.read_parquet(RESOURCES.joinpath(name))

        ds = Dataset(df)
        datasets.append(ds)

    if len(datasets) == 1:
        return datasets[0]

    return tuple(datasets)
