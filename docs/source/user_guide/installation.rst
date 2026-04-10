Installation
============
This page will guide you through the installation process for `ampworks`. The software is currently only available on `PyPI <https://pypi.org/project/ampworks>`_ (for stable releases) and `GitHub <https://github.com/NatLabRockies/ampworks>`_ (for stable and developer releases).

Installing via PyPI
-------------------
Installing with `pip` will pull a distribution file from the Python Package Index (PyPI). We provide both binary and source distributions on `PyPI <https://pypi.org/project/ampworks>`_.

To install the latest release, simply run the following::

    pip install ampworks[gui]

Note that the `[gui]` extra is required to install the graphical user interface (GUI) dependencies. These are entirely optional, as you can use any `ampworks` functionality without the GUIs as well. We leave the choice up to you. Leaving out the `[gui]` extra will install fewer dependencies in your environment and will generally resolver and install faster, but you will not have access to the GUIs.

Python Version Support
----------------------
Please note that `ampworks` releases only support whichever Python versions are actively maintained at the time of the release. If you are using a version of Python that has reached the end of its life, as listed on the `official Python release page`_, you may need to install an older version of `ampworks` or upgrade your Python version. To install a specific, older version that supports your current Python installation, use::

    pip install ampworks[gui]==x.x

where `x.x` is replaced with a specific major/minor version number. We recommend, however, upgrading your Python version instead of using an older version of `ampworks` if you are able to do so. As mentioned above, the `[gui]` extra is optional, so you can leave it out if you do not plan to use the GUI features, and would prefer not to install the additional dependencies.

.. _official Python release page: https://devguide.python.org/versions/

Developer Versions
------------------
The development version is ONLY hosted on GitHub. To install it, see the :doc:`/development/index` section. You should only do this if you:

* Want to try experimental features
* Need access to unreleased fixes
* Would like to contribute to the package