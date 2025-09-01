# Changelog

## v0.1.2 – 2025-09-01

**New Features**:

- `TimeseriesExtractor` class for downloading per-feature CSV time series from Earth Engine `ee.ImageCollection`.
  - Point geometries: uses `getRegion`, no reducer required.
  - Polygon/MultiPolygon geometries: supports reducers for aggregation.
  - Exports directly to CSV for downstream analysis.
- `HarmonicRegression` class for performing harmonic regression on Earth Engine `ee.ImageCollection`.
  - Computes harmonic coefficients with configurable order and base frequency.
  - Provides phase and amplitude outputs for phenology and seasonal dynamics analysis.
  - Supports fitted harmonic time series generation for visualization or modeling.

**Improvements**:

- Updated docstrings for clarity and alignment with function behavior.
- Documentation expanded to include harmonic regression and time series extraction workflows.

---

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
