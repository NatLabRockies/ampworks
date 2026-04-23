"""
The `_core` subpackage hosts all of the functions, classes, etc. that should be
made available at the base-level of the package.

"""

from ._dataset import Dataset
from ._headers import HeaderAliases, standardize_headers
from ._read import read_csv, read_excel, read_table

__all__ = [
    'Dataset',
    'read_csv',
    'read_excel',
    'read_table',
    'HeaderAliases',
    'standardize_headers',
]
