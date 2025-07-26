
# timeseries module

geeagri.timeseries provides tools for extracting and analyzing time series data from Google Earth Engine. Below are the main functions and classes:

## extract_timeseries_to_point

Extracts pixel time series from an ee.ImageCollection at a point.

**Example:**

```python
import geeagri
import ee
ee.Authenticate()
ee.Initialize()

lat, lon = 6.5244, 3.3792
collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
df = geeagri.extract_timeseries_to_point(
    lat, lon, collection, start_date='2020-01-01', end_date='2020-12-31', band_names=['NDVI'], scale=30
)
print(df.head())
```

## extract_timeseries_to_polygon

Extracts time series statistics over a polygon from an ee.ImageCollection.


**Example:**

```python
polygon = ee.Geometry.Polygon([
    [[3.37, 6.52], [3.38, 6.52], [3.38, 6.53], [3.37, 6.53], [3.37, 6.52]]
])
df_poly = geeagri.extract_timeseries_to_polygon(
    polygon, collection, start_date='2020-01-01', end_date='2020-12-31', band_names=['NDVI'], scale=30, reducer="MEAN"
)
print(df_poly.head())
```

## HarmonicRegression

Class for performing harmonic regression on an Earth Engine ImageCollection.


**Example:**

```python
from geeagri.timeseries import HarmonicRegression

ref_date = '2020-01-01'
order = 2
hr = HarmonicRegression(collection, ref_date, 'NDVI', order=order)
harmonic_coeffs = hr.get_harmonic_coeffs()
fitted = hr.get_fitted_harmonics(harmonic_coeffs)
```

See the example notebooks for more advanced workflows and visualizations.
