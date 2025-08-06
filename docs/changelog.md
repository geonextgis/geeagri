# Changelog

## v0.1.1 – 2025-08-06

**New Features**:

- `ImagePatchExtractor` class added for efficient extraction of image patches from Earth Engine `ee.Image` objects using local sample points as `GeoDataFrame`.
- Supports multiple export formats: `png`, `jpg`, `GEO_TIFF`, and others.
- Fully parallelized with configurable number of processes.
- Uses a specified identifier column for naming output files.
- Automatically handles patch sizing via `dimensions` and `buffer` parameters.

**Improvements**
- Improved documentation and example notebooks:

---

## v0.1.0 - 2025-07-29

**Improvements**:

- Improved initial project scaffolding and modular structure.
- Enhanced configuration for easier customization and extension.

**New Features**:

- Added preprocessing module with various image scaling options:
  - MeanCentering
  - MinMaxScaler
  - StandardScaler
  - RobustScaler
- Added analysis module including easy implementation of PCA with explained variance calculation.
- Added new example notebooks.
