# Welcome to geeagri

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geonextgis/geeagri/blob/main)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geonextgis/geeagri/main?labpath=notebooks%2Fintro.ipynb)
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/geonextgis/geeagri/blob/main/notebooks/intro.ipynb)
[![PyPI Version](https://img.shields.io/pypi/v/geeagri.svg)](https://pypi.org/project/geeagri)
[![Downloads](https://static.pepy.tech/badge/geeagri)](https://pepy.tech/project/geeagri)
[![Documentation Status](https://github.com/geonextgis/geeagri/workflows/docs/badge.svg)](https://geonextgis.github.io/geeagri)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <a href="https://geonextgis.github.io/geeagri">
    <img src="https://raw.githubusercontent.com/geonextgis/geeagri/main/docs/assets/logo.png" alt="logo" width="200"/>
  </a>
</div>

**A Python package for agricultural monitoring and analysis using Google Earth Engine**

- GitHub repo: <https://github.com/geonextgis/geeagri>
- Documentation: <https://geonextgis.github.io/geeagri>
- PyPI: <https://pypi.org/project/geeagri>
- Notebooks: <https://github.com/geonextgis/geeagri/tree/main/docs/examples>
- License: [MIT](https://opensource.org/licenses/MIT)

---

## Introduction

**geeagri** is a Python package that integrates the power of [Google Earth Engine (GEE)](https://earthengine.google.com/) with domain-specific agricultural analysis. It enables scalable processing, downloading, and visualization of satellite data for crop monitoring, yield estimation, and agro-environmental assessment.

This package builds upon geospatial tools like [`geemap`](https://github.com/gee-community/geemap) and simplifies workflows for scientists, researchers, and policymakers working in agriculture. Whether you're interested in vegetation monitoring, drought assessment, phenology extraction, or productivity mapping, **geeagri** offers tools and prebuilt pipelines to make your analysis easier and faster.

**geeagri** is ideal for:
- Researchers working with satellite-derived agricultural indicators.
- Practitioners and analysts from development, environmental, and governmental organizations.
- Students and educators looking to learn remote sensing for agriculture.

For a complete list of examples and use cases, visit the [notebooks](https://github.com/geonextgis/geeagri/tree/main/docs/examples) section.

---

## Key Features

- Extracting long-term time series data from GEE for both point and polygon geometries.
- Easy implementation of Harmonic Regression on vegetation or climate indices.

---