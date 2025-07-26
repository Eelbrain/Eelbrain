# Eelbrain

[![DOI](https://zenodo.org/badge/3651023.svg)](https://zenodo.org/badge/latestdoi/3651023)

## Latest version

| Stable | Pre-release |
| ------ | ----------- |
| [![Conda Version](https://img.shields.io/conda/vn/conda-forge/eelbrain)](https://anaconda.org/conda-forge/eelbrain) | [![Anaconda-Server Badge](https://anaconda.org/conda-forge/eelbrain/badges/version.svg)](https://anaconda.org/conda-forge/eelbrain/labels) |

## Resources

- Documentation: http://eelbrain.readthedocs.io
- Wiki: https://github.com/christianbrodbeck/Eelbrain/wiki
- GitHub: https://github.com/christianbrodbeck/Eelbrain
- Conda-forge feedstock: https://github.com/conda-forge/eelbrain-feedstock

## Testing

To expedite testing in different environments, there are several shortcut commands (from the `Eelbrain` root directory):

- `$ make test-no-gui`: runs all tests that do not invoke a GUI
- `$ make test-only-gui`: runs specifically those tests that involve GUI elements
- `pytest --runslow eelbrain/_experiment/tests/test_sample_experiment.py::test_sample_source` runs more thorough (and lengthy) testing of the `MneExperiment` pipeline 
