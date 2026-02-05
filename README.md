# ampworks

[![CI][ci-b]][ci-l] &nbsp;
![tests][test-b] &nbsp;
![coverage][cov-b] &nbsp;
[![ruff][ruff-b]][ruff-l] &nbsp;
[![pep8][pep-b]][pep-l]

[ci-b]: https://github.com/NatLabRockies/ampworks/actions/workflows/ci.yml/badge.svg
[ci-l]: https://github.com/NatLabRockies/ampworks/actions/workflows/ci.yml

[test-b]: https://github.com/NatLabRockies/ampworks/blob/main/images/tests.svg?raw=true
[cov-b]: https://github.com/NatLabRockies/ampworks/blob/main/images/coverage.svg?raw=true

[ruff-b]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-l]: https://github.com/astral-sh/ruff

[pep-b]: https://img.shields.io/badge/code%20style-pep8-orange.svg
[pep-l]: https://www.python.org/dev/peps/pep-0008

## Summary
`ampworks` is a collection of tools designed to visualize and process experimental battery data. It provides routines for degradation mode analysis, parameter extraction from common protocols (e.g., GITT, ICI, etc.), and more. These routines provide key properties for life and physics-based models (e.g., SPM and P2D). Graphical user interfaces (GUIs) are available for some of the analyses. See a full list of the GUI-based applications by running `ampworks -h` in your terminal after installation. 

Note: `ampworks` is in early development. The API may change as it matures.

## Installation
`ampworks` can be installed from [PyPI](https://pypi.org/project/ampworks) using the following command:

```
pip install ampworks[gui]
```

Using `[gui]` is optional. When included, the installation includes extra dependencies needed for the GUI-based applications. However, the GUIs are completely optional. Any routines that can accessed through a GUI can also be implemented in scripts or notebooks. The package will generally install faster without the extra dependencies. Note that you can always add the GUI dependencies at a later time too; they do not need to be included with the original installation of `ampworks`.

For those interested in setting up a developer and/or editable version of this software please see the directions available in the "Development" section of our [documentation](https://ampworks.readthedocs.io/en/latest/development).

## Get Started
The best way to get started is by exploring the `examples` folder, which includes real datasets and demonstrates key functionality. These examples will evolve as the software progresses.

**Notes:**
* If you are new to Python, check out [Spyder IDE](https://www.spyder-ide.org/). Spyder is a powerful interactive development environment (IDE) that can make programming in Python more approachable to new users.
* Another friendly option for getting started in Python is to use [Jupyter Notebooks](https://jupyter.org/). We write our examples in Jupyter Notebooks since they support both markdown blocks for explanations and executable code blocks.
* Python, Spyder, and Jupyter Notebooks can be setup using [Anaconda](https://www.anaconda.com/download/success). Anaconda provides a convenient way for new users to get started with Python due to its friendly graphical installer and environment manager.

## Citing this Work
This work was authored by researchers at the National Laboratory of the Rockies (NLR). If you use use this package in your work, please include the following citation:

> Randall, Corey R. "ampworks: Battery data analysis tools in Python [SWR-25-39]." Computer software, Mar. 2025. url: [github.com/NatLabRockies/ampworks](https://github.com/NatLabRockies/ampworks). doi: [10.11578/dc.20250313.2](https://doi.org/10.11578/dc.20250313.2).

For convenience, we also provide the following for your BibTex:

```
@misc{Randall-2025,
  author = {Randall, Corey R.},
  title = {{ampworks: Battery data analysis tools in Python [SWR-25-39]}},
  url = {github.com/NatLabRockies/ampworks},
  month = {Mar.},
  year = {2025},
  doi = {10.11578/dc.20250313.2},
}
```

## Contributing
If you'd like to contribute to this package, please look through the existing [issues](https://github.com/NatLabRockies/ampworks/issues). If the bug you've caught or the feature you'd like to add isn't already being worked on, please submit a new issue before getting started. You should also read through the [developer guidelines](https://ampworks.readthedocs.io/en/latest/development).

## Disclaimer
This work was authored by the National Laboratory of the Rockies (NLR), operated by Alliance for Energy Innovation, LLC, for the U.S. Department of Energy (DOE). The views expressed in the repository do not necessarily represent the views of the DOE or the U.S. Government.
