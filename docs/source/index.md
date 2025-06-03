# Welcome to ONTraC (Ordered Niche Trajectory Construction)

<span>
  <a href="https://pypi.org/project/ONTraC/">
    <img src="https://img.shields.io/pypi/v/ONTraC.svg" alt="PyPI version" style="display:inline-block;">
  </a>
  <a href="https://pypi.org/project/ONTraC/">
    <img src="https://img.shields.io/pypi/pyversions/ONTraC.svg" alt="Python versions" style="display:inline-block;">
  </a>
  <a href="https://pepy.tech/project/ONTraC">
    <img src="https://static.pepy.tech/badge/ONTraC" alt="Downloads" style="display:inline-block;">
  </a>
  <a href="https://pypi.org/project/ONTraC/">
    <img src="https://img.shields.io/pypi/dm/ONTraC.svg" alt="PyPI Downloads" style="display:inline-block;">
  </a>
  <a href="https://anaconda.org/gyuanlab/ontrac">
    <img src="https://anaconda.org/gyuanlab/ontrac/badges/version.svg" alt="Anaconda-Server Version" style="display:inline-block;">
  </a>
  <a href="https://anaconda.org/gyuanlab/ontrac">
    <img src="https://anaconda.org/gyuanlab/ontrac/badges/platforms.svg" alt="Anaconda-Server Platforms" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC">
    <img src="https://badgen.net/github/stars/gyuanlab/ONTraC" alt="GitHub Stars" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC/issues">
    <img src="https://img.shields.io/github/issues/gyuanlab/ONTraC.svg" alt="GitHub Issues" style="display:inline-block;">
  </a>
  <a href="https://github.com/gyuanlab/ONTraC/blob/master/LICENSE">
    <img src="https://img.shields.io/github/license/gyuanlab/ONTraC.svg" alt="GitHub License" style="display:inline-block;">
  </a>
</span>

## Overview

ONTraC (Ordered Niche Trajectory Construction) is a niche-centered, machine learning method for constructing spatially continuous trajectories.

ONTraC differs from existing tools in that it treats a niche, rather than an individual cell, as the basic unit for spatial trajectory analysis. In this context, we define niche as a multicellular, spatially localized region where different cell types may coexist and interact with each other.

ONTraC seamlessly integrates niche features (cell-type composition) and spatial information by using the graph neural network modeling framework. Its output, which is called the niche trajectory, can be viewed as a one dimensional representation of the tissue microenvironment continuum. By disentangling cell-level and niche-level properties, niche trajectory analysis provides a coherent framework to study coordinated responses from all the cells in association with continuous tissue microenvironment variations.

For installation instructions, see [Installation and Setup](./installation.md).
For detailed explanations of key concepts like niches and niche trajectoryies, see [Key Concepts and Terminology].

```{image} ../source/images/other/logo_with_text_long.png
:alt: ONTraC logo
```

```{image} ../source/images/other/ONTraC_structure.png
:alt: ONTraC structure
```

Check out the [installation](./installation.md) for installation guidelines.

Check out the [usage](./usage.md) for details if you want to use ONTraC with command lines.

Check out the [step-by-step tutorial](./step_by_step_tutorial.ipynb) for details if you want to use ONTraC within a Jupyter notebook.

```{toctree}
:hidden: true

installation
usage
tutorials
examples
contributors
citation
```

```{note}
This project is under active development.
```

## Interoperability

ONTraC has been incorporated with [Giotto Suite](https://drieslab.github.io/Giotto_website/articles/ontrac.html).

## Citation

**Wang, W.\*, Zheng, S.\*, Shin, C. S., Chávez-Fuentes J. C.  & [Yuan, G. C.](https://labs.icahn.mssm.edu/yuanlab/)$**. [ONTraC characterizes spatially continuous variations of tissue microenvironment through niche trajectory analysi](https://doi.org/10.1186/s13059-025-03588-5). *Genome Biol*, 2025.

## Other Resources

[Reproducible codes](https://github.com/gyuanlab/ONTraC_paper)

[Dataset used in our paper](https://doi.org/10.5281/zenodo.11186619)
