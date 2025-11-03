---
title: 'geeagri: A Python package for agricultural analysis with Google Earth Engine'
tags:
  - Python
  - Google Earth Engine
  - agriculture
  - remote sensing
  - analysis
authors:
  - name: Krishnagopal Halder
    orcid: 0009-0005-9815-3017
    corresponding: true
    affiliation: 1
  - name: Amit Kumar Srivastava
    affiliation: 1
  - name: Manmeet Singh
    affiliation: 2
  - name: Frank Ewert
    orcid: 0000-0002-4392-8154
    affiliation: "1, 3"
affiliations:
 - name: Leibniz Centre for Agricultural Landscape Research (ZALF), Eberswalder Str. 84, 15374 Müncheberg, Germany
   index: 1
 - name: Department of Earth and Planetary Sciences, Jackson School of Geosciences, University of Texas, Austin, USA
   index: 2
 - name: Institute of Crop Science and Resource Conservation, University of Bonn, Katzenburgweg 5, D-53115, Bonn, Germany
   index: 3
date: 02 November 2025
bibliography: paper.bib
---

# Summary

`geeagri` is a Python package designed for advanced agricultural monitoring and analysis using Google Earth Engine (GEE). GEE is a powerful cloud-computing platform with a multi-petabyte catalog of satellite imagery and geospatial datasets [@gorelick2017google]. Although GEE provides its robust JavaScript and Python APIs, and packages like `geemap` have improved its interactive mapping capabilities [@wu2020geemap], researchers in domain-specific fields like agriculture still face significant challenges. Tasks such as crop modeling [@ines2013assimilation] and machine learning [@zhu2017deep] often demand complex, repetitive workflows for geospatial data collection and extensive pre-processing before analysis.

`geeagri` addresses these challenges by providing high-level, user-friendly workflows for agricultural analysis. It encapsulates data collection and processing pipelines within an object-oriented framework, enabling both simple and advanced analyses to be performed seamlessly on GEE. By automating repetitive tasks and standardizing data processing procedures, `geeagri` facilitates the work of crop modelers, remote sensing analysts, and machine learning researchers, thereby reducing computational overhead on local systems and minimizing the need for extensive manual scripting.

# Statement of need

Agricultural research increasingly relies on satellite-derived and climate data for applications such as crop modeling, data assimilation, yield forecasting, and vegetation monitoring [@ines2013assimilation]. While Google Earth Engine (GEE) offers vast geospatial datasets and powerful cloud-based computation [@gorelick2017google], its standard JavaScript and Python APIs require extensive manual scripting to perform domain-specific tasks. Existing tools like `geemap` enhance visualization and general geospatial analysis [@wu2020geemap], but they do not fully address the specialized needs of agricultural workflows.

`geeagri` addresses this gap by providing high-level, domain-focused workflows for agricultural analysis within GEE. It streamlines tasks such as large scale data extraction (tabular or imagery), cloud masking, curve fitting, and other complex processing pipelines through an object-oriented and modular framework. The package's functionality is targeted at specific user groups:
- **Crop modelers**, who require high-temporal-resolution climate data (e.g., temperature, precipitation, solar radiation) and phenological indicators at site or regional scales [@ines2013assimilation].
- **Remote sensing analysts**, who need seamless workflows for data extraction and data preprocessing tasks such as cloud masking and advanced analyses including Principal Component Analysis and Harmonic Regression [@beck2006improved].
- **Machine learning researchers** training vision models on large number of tile-based image data from GEE [@zhu2017deep].

## Key Features

* **Parallelized Data Extraction:** Automates the retrieval of satellite time series and climate datasets for any point or polygon location. Supports efficient parallel downloading of image chips (e.g., Sentinel-2 [@drusch2012sentinel], Landsat [@wulder2016global]) for large-scale machine learning applications.

* **Automated Cloud Masking and Filtering:** Provides user-defined cloud thresholding and masking with minimal coding effort.

* **Synthetic Time Series Generation:** Creates temporally consistent datasets (e.g., 5-day or 10-day intervals) from irregular satellite observations to support time-series analyses [@chen2004simple].

* **Phenological Analysis:** Performs curve fitting and extracts key land surface phenology metrics, including the start (SOS), end (EOS), and peak (POS) of the growing season [@jonsson2004timesat].

* **High-Level Workflow Abstraction:** Encapsulates multi-step GEE processes into simple functions and object-oriented classes, enabling reproducible and streamlined agricultural data workflows.

# Example Use

The example below demonstrates how `geeagri` can be used to extract daily climate time series (temperature, precipitation, and solar radiation) from the ERA5-Land dataset for a specific point location (see \autoref{fig:figure}).

```python
import ee
import geeagri
from geeagri.extract import extract_timeseries_to_point

# Authenticate and initialize the Earth Engine API
ee.Authenticate()
ee.Initialize()

# Define point location (longitude, latitude)
lon, lat = -98.15, 30.50
point = ee.Geometry.Point([lon, lat])

# Load ERA5-Land daily aggregated climate dataset
era5_land = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")

# Extract daily temperature, precipitation, and solar radiation time series
era5_land_point_ts = extract_timeseries_to_point(
    lat=lat,
    lon=lon,
    image_collection=era5_land,
    start_date="2020-01-01",
    end_date="2021-01-01",
    band_names=[
        "temperature_2m_min",
        "temperature_2m_max",
        "total_precipitation_sum",
        "surface_solar_radiation_downwards_sum",
    ],
    scale=11132,  # spatial resolution in meters (~11 km)
)
```

![Example of extracting a daily climate time series from ERA5-Land for a single point using geeagri. The plot shows daily maximum and minimum temperature values for 2020, illustrating how the extracted data can support crop model simulations and environmental analyses.\label{fig}](figure.png)

## `geeagri` Tutorials

Comprehensive tutorials, example notebooks, and API documentation are available to help users get started with `geeagri` and explore its functionality:

* [**Example Notebooks**](https://github.com/geonextgis/geeagri/tree/main/docs/examples): step-by-step demonstrations of common workflows.
* [**API Documentation and Website**](https://geonextgis.github.io/geeagri/): detailed reference of available modules, classes, and functions, with usage guidelines and examples.

# Acknowledgements

We gratefully acknowledge the foundational work of the Google Earth Engine team [@gorelick2017google] and the `geemap` package [@wu2020geemap], which form the basis of `geeagri`. This work has been supported by the German Federal Ministry of Education and Research (BMBF) through the projects LL-SYSTAIN (Grant Labs-2024-IÖR) and FAIRagro (Grant NFDI 51/1), and by the Biotechnology and Biological Sciences Research Council (BBSRC) through the project SynPAI (Grant BB/S020969).

# References
