
# Usage

geeagri provides utilities for extracting, analyzing, and visualizing time series data from Google Earth Engine. Below are some common usage examples.


## 1. Importing geeagri

```python
import geeagri
```

ee.Initialize()
df = geeagri.extract_timeseries_to_point(

## 2. Extracting a Time Series at a Point

```python
import ee
ee.Authenticate()
ee.Initialize()

lat, lon = 6.5244, 3.3792  # Example: Lagos, Nigeria
collection = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
band = 'NDVI'
start_date = '2020-01-01'
end_date = '2020-12-31'
scale = 30
df = geeagri.extract_timeseries_to_point(
    lat, lon, collection, start_date, end_date, band_names=[band], scale=scale
)
print(df.head())
```

df_poly = geeagri.extract_timeseries_to_polygon(

## 3. Extracting a Time Series for a Polygon

```python
polygon = ee.Geometry.Polygon([
    [[3.37, 6.52], [3.38, 6.52], [3.38, 6.53], [3.37, 6.53], [3.37, 6.52]]
])
df_poly = geeagri.extract_timeseries_to_polygon(
    polygon, collection, start_date, end_date, band_names=[band], scale=scale, reducer="MEAN"
)
print(df_poly.head())
```


## 4. Harmonic Regression Example

```python
from geeagri.timeseries import HarmonicRegression

ref_date = start_date
order = 2
hr = HarmonicRegression(collection, ref_date, band, order=order)
harmonic_coeffs = hr.get_harmonic_coeffs()
# You can now use hr.get_fitted_harmonics(harmonic_coeffs) for fitted values
```

df.plot(x='time', y=band, marker='o')

## 5. Plotting the Results

```python
import matplotlib.pyplot as plt
df.plot(x='time', y=band, marker='o')
plt.title('Time Series at Point')
plt.show()
```
