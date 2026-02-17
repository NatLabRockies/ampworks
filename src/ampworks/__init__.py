"""
Summary
=======
`ampworks` is a collection of tools designed to visualize and process
experimental battery data. It provides routines for degradation mode analysis,
parameter extraction from common protocols (e.g., GITT, ICI, etc.), and more.
These routines provide key properties for life and physics-based models (e.g.,
SPM and P2D). Graphical user interfaces (GUIs) are available for some of the
analyses. See a list of the GUI-based applications by running `ampworks -h`
in your terminal after installation.

Note: `ampworks` is in early development. The API may change as it matures.

Accessing the Documentation
---------------------------
Documentation is accessible via Python's `help()` function which prints
docstrings from a package, module, function, class, etc. You can also access
the documentation by visiting the website, hosted on Read the Docs. The website
includes search functionality and more detailed examples.

"""

from typing import TYPE_CHECKING

from ._core import (
    Dataset,
    read_csv,
    read_excel,
    read_table,
)

__version__ = '0.0.2.dev0'

__all__ = [
    'Dataset',
    'read_csv',
    'read_excel',
    'read_table',
    'ocv',
    'ici',
    'gitt',
    'dqdv',
    'hppc',
    'utils',
    'datasets',
    'mathutils',
    'plotutils',
    '_in_interactive',
    '_in_notebook',
]

if TYPE_CHECKING:  # pragma: no cover
    from ampworks import (
        ocv, ici, gitt, dqdv, hppc, utils, datasets, mathutils, plotutils,
    )


# Lazily load submodules/subpackages
_lazy_modules = {
    'ocv': 'ampworks.ocv',
    'ici': 'ampworks.ici',
    'gitt': 'ampworks.gitt',
    'dqdv': 'ampworks.dqdv',
    'hppc': 'ampworks.hppc',
    'utils': 'ampworks.utils',
    'datasets': 'ampworks.datasets',
    'mathutils': 'ampworks.mathutils',
    'plotutils': 'ampworks.plotutils',
}


def __getattr__(name):
    import importlib

    if name in _lazy_modules:
        module = importlib.import_module(_lazy_modules[name])
        globals()[name] = module  # cache for later
        return module

    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    return list(globals()) + list(_lazy_modules)


# Check for interactive and notebook environments
def _in_interactive() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        return shell in ('ZMQInteractiveShell',)  # Jupyter Notebook or Lab
    except Exception:
        return False
