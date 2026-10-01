<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="snowagg/docs/img/logo-dark.svg">
    <img src="snowagg/docs/img/logo-light.svg" alt="snowagg" width="420">
  </picture>
</p>

# aggregation, with snowagg

This repository is a fork of **[aggregation](https://github.com/jleinonen/aggregation)**
by Jussi Leinonen, in the version of the
**[OPTIMICe-team fork](https://github.com/OPTIMICe-team/aggregation)**, with its full
history (up to commit `f05c607`). The original Python package is in `aggregation/`,
unchanged except for one fix for SciPy ≥ 1.14 (`cumtrapz` → `cumulative_trapezoid` in
`rotator.py`).

This fork adds **[snowagg](snowagg/)**: a C++ implementation of the same model with the
same Python API, about 20–90 times faster for complete runs. **snowagg is completely
vibe coded** (written by an AI coding assistant) and verified against the original
package in this repository; see **[snowagg/README.md](snowagg/README.md)** for its
installation, verification, figures and differences to the original.

All credit for the model goes to the authors of the original. Its README follows.

---

A Python code for generating 3D models of aggregate 
and rimed snowflakes.

Requires NumPy and SciPy.

You can install the package by running 
```
python setup.py install
```
in the repository root directory.

A more thorough documentation will follow later. Currently, you can 
find some examples in the notebook at [notebooks/Rimed_aggregates.ipynb](notebooks/Rimed_aggregates.ipynb). You can view this by going to the notebooks directory and running
```
jupyter notebook Rimed_aggregates.ipynb
```
(you need jupyter installed to do this).
