# ampworks Changelog

## [Unreleased](https://github.com/NatLabRockies/ampworks)

### New Features
- New `ocv` module with `match_peaks` function and supporting `DqdvSpline` ([#14](https://github.com/NatLabRockies/ampworks/pull/14))
- Add `hppc` subpackage to extract impedance from HPPC protocols ([#12](https://github.com/NatLabRockies/ampworks/pull/12))
- Update dQdV GUI figure to gridspec with electrode voltages ([#10](https://github.com/NatLabRockies/ampworks/pull/10))
- New `RichTable` container with column validation ([#9](https://github.com/NatLabRockies/ampworks/pull/9))
- Include standard deviations in dQdV fits ([#8](https://github.com/NatLabRockies/ampworks/pull/8))
- Add `utils` subpackage with a handful of basic utilities ([#7](https://github.com/NatLabRockies/ampworks/pull/7))
- New `ici` and `datasets` modules, and `Dataset` class... needs tests ([#4](https://github.com/NatLabRockies/ampworks/pull/4))
- Add version warning banner to docs for dev and older releases ([#3](https://github.com/NatLabRockies/ampworks/pull/3))
- Complete overhaul to `plotutils` for shorter, modular use ([#2](https://github.com/NatLabRockies/ampworks/pull/2))

### Optimizations
- Store custom plotly template and config in `_style` module ([#10](https://github.com/NatLabRockies/ampworks/pull/10))
- Added tests for `gitt.extract_params` ([#6](https://github.com/NatLabRockies/ampworks/pull/6))
- Added tests for `ici.extract_params` ([#5](https://github.com/NatLabRockies/ampworks/pull/5))

### Bug Fixes
- Catch when `NaN` is present when parsing for headers in excel files ([#23](https://github.com/NatLabRockies/ampworks/pull/23))
- Force the `cell` dataset of the `DqdvFitter` to require an Ah column ([#19](https://github.com/NatLabRockies/ampworks/pull/19))
- Update patching policy for releases, use `spellcheck` in nox pre-commit ([#13](https://github.com/NatLabRockies/ampworks/pull/13))
- Readers missing name-only columns, e.g., `testtime` ([#8](https://github.com/NatLabRockies/ampworks/pull/8))

### Breaking Changes
- Complete overhaul to `plotutils` for shorter, modular use ([#2](https://github.com/NatLabRockies/ampworks/pull/2))

### Chores
- Enable and configure `codecov` for tracking test coverage and comparing PRs ([#21](https://github.com/NatLabRockies/ampworks/pull/21))
- Organize example datasets by module to improve discovery and scalability ([#20](https://github.com/NatLabRockies/ampworks/pull/20))
- Remove `flake8` and `autopep8` and use `ruff` for linting/formatting instead ([#18](https://github.com/NatLabRockies/ampworks/pull/18))
- Make GitHub hyperlinks reference new org name `NREL` -> `NatLabRockies` ([#17](https://github.com/NatLabRockies/ampworks/pull/17))
- Add project overview page to development section in documentation ([#16](https://github.com/NatLabRockies/ampworks/pull/16)) 
- Rebrand NREL to NLR, and include name change for Alliance as well ([#15](https://github.com/NatLabRockies/ampworks/pull/15))

## [v0.0.1](https://github.com/NatLabRockies/ampworks/tree/v0.0.1)
This is the first release of `ampworks`. Main features/capabilities are listed below.

### Features
- Processing routines for degradation mode analysis
- Functions to extract parameters from GITT data

### Notes
- Still in development, API likely to change as software matures
- Documentation available on [Read the Docs](https://ampworks.readthedocs.io/)
