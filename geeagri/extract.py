"""Module for extracting data from Google Earth Engine."""

import math
import ee
import pandas as pd
import geopandas as gpd
import os
import logging
import json
from datetime import datetime
import multiprocessing
from retry import retry
import requests
import shutil
from typing import Union, Optional, Dict, List
from pathlib import Path


def extract_timeseries_to_point(
    lat,
    lon,
    image_collection,
    start_date=None,
    end_date=None,
    band_names=None,
    scale=None,
    crs=None,
    crsTransform=None,
    out_csv=None,
):
    """
    Extracts pixel time series from an ee.ImageCollection at a point.

    Args:
        lat (float): Latitude of the point.
        lon (float): Longitude of the point.
        image_collection (ee.ImageCollection): Image collection to sample.
        start_date (str, optional): Start date (e.g., '2020-01-01').
        end_date (str, optional): End date (e.g., '2020-12-31').
        band_names (list, optional): List of bands to extract.
        scale (float, optional): Sampling scale in meters.
        crs (str, optional): Projection CRS. Defaults to image CRS.
        crsTransform (list, optional): CRS transform matrix (3x2 row-major). Overrides scale.
        out_csv (str, optional): File path to save CSV. If None, returns a DataFrame.

    Returns:
        pd.DataFrame or None: Time series data if not exporting to CSV.
    """

    if not isinstance(image_collection, ee.ImageCollection):
        raise ValueError("image_collection must be an instance of ee.ImageCollection.")

    property_names = image_collection.first().propertyNames().getInfo()
    if "system:time_start" not in property_names:
        raise ValueError("The image collection lacks the 'system:time_start' property.")

    point = ee.Geometry.Point([lon, lat])

    try:
        if start_date and end_date:
            image_collection = image_collection.filterDate(start_date, end_date)
        if band_names:
            image_collection = image_collection.select(band_names)
        image_collection = image_collection.filterBounds(point)
    except Exception as e:
        raise RuntimeError(f"Error filtering image collection: {e}")

    try:
        result = image_collection.getRegion(
            geometry=point, scale=scale, crs=crs, crsTransform=crsTransform
        ).getInfo()

        result_df = pd.DataFrame(result[1:], columns=result[0])

        if result_df.empty:
            raise ValueError(
                "Extraction returned an empty DataFrame. Check your point, date range, or selected bands."
            )

        result_df["time"] = result_df["time"].apply(
            lambda t: datetime.utcfromtimestamp(t / 1000)
        )

        if out_csv:
            result_df.to_csv(out_csv, index=False)
        else:
            return result_df

    except Exception as e:
        raise RuntimeError(f"Error extracting data: {e}.")


def extract_timeseries_to_polygon(
    polygon,
    image_collection,
    start_date=None,
    end_date=None,
    band_names=None,
    scale=None,
    crs=None,
    reducer="MEAN",
    out_csv=None,
):
    """
    Extracts time series statistics over a polygon from an ee.ImageCollection.

    Args:
        polygon (ee.Geometry.Polygon): Polygon geometry.
        image_collection (ee.ImageCollection): Image collection to sample.
        start_date (str, optional): Start date (e.g., '2020-01-01').
        end_date (str, optional): End date (e.g., '2020-12-31').
        band_names (list, optional): List of bands to extract.
        scale (float, optional): Sampling scale in meters.
        crs (str, optional): Projection CRS. Defaults to image CRS.
        reducer (str or ee.Reducer): Name of reducer or ee.Reducer instance.
        out_csv (str, optional): File path to save CSV. If None, returns a DataFrame.

    Returns:
        pd.DataFrame or None: Time series data if not exporting to CSV.
    """

    if not isinstance(image_collection, ee.ImageCollection):
        raise ValueError("image_collection must be an instance of ee.ImageCollection.")
    if not isinstance(polygon, ee.Geometry):
        raise ValueError("polygon must be an instance of ee.Geometry.")

    # Allowed reducers
    allowed_statistics = {
        "COUNT": ee.Reducer.count(),
        "MEAN": ee.Reducer.mean(),
        "MEAN_UNWEIGHTED": ee.Reducer.mean().unweighted(),
        "MAXIMUM": ee.Reducer.max(),
        "MEDIAN": ee.Reducer.median(),
        "MINIMUM": ee.Reducer.min(),
        "MODE": ee.Reducer.mode(),
        "STD": ee.Reducer.stdDev(),
        "MIN_MAX": ee.Reducer.minMax(),
        "SUM": ee.Reducer.sum(),
        "VARIANCE": ee.Reducer.variance(),
    }

    # Get reducer from string or use directly
    if isinstance(reducer, str):
        reducer_upper = reducer.upper()
        if reducer_upper not in allowed_statistics:
            raise ValueError(
                f"Reducer '{reducer}' not supported. Choose from: {list(allowed_statistics.keys())}"
            )
        reducer = allowed_statistics[reducer_upper]
    elif not isinstance(reducer, ee.Reducer):
        raise ValueError("reducer must be a string or an ee.Reducer instance.")

    # Filter dates and bands
    if start_date and end_date:
        image_collection = image_collection.filterDate(start_date, end_date)
    if band_names:
        image_collection = image_collection.select(band_names)

    image_collection = image_collection.filterBounds(polygon)

    def image_to_dict(image):
        date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd")
        stats = image.reduceRegion(
            reducer=reducer, geometry=polygon, scale=scale, crs=crs, maxPixels=1e13
        )
        return ee.Feature(None, stats).set("time", date)

    stats_fc = image_collection.map(image_to_dict).filter(
        ee.Filter.notNull(image_collection.first().bandNames())
    )

    try:
        stats_list = stats_fc.getInfo()["features"]
    except Exception as e:
        raise RuntimeError(f"Error retrieving data from GEE: {e}")

    if not stats_list:
        raise ValueError("No data returned for the given polygon and parameters.")

    records = []
    for f in stats_list:
        props = f["properties"]
        records.append(props)

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"])
    df.insert(0, "time", df.pop("time"))

    if out_csv:
        df.to_csv(out_csv, index=False)
    else:
        return df


class TimeseriesExtractor:
    """
    Downloads per-feature CSV time series from an Earth Engine ImageCollection.

    - If a feature geometry is a Point: uses getRegion; no reducer needed.
    - If a feature geometry is a Polygon/MultiPolygon: requires a reducer.

    Args:
        image_collection (ee.ImageCollection)
        sample_gdf (gpd.GeoDataFrame): Must contain geometry and `identifier` column.
        identifier (str): Column name to use for file naming (also added to CSV).
        out_dir (str): Output directory (created if absent).
        selectors (list|None): Optional property list to include in CSV export.
                               (If provided, 'time' will be auto-added if missing.)
        scale (int|None): Pixel scale (meters) for sampling/reduction.
        crs (str|None): Projection CRS. Default 'EPSG:4326'.
        crsTransform (list|None): 3x2 transform for getRegion (points) if you need it.
        num_processes (int): Parallel workers.
        start_date (str|None): 'YYYY-MM-DD'. If provided with end_date, filters IC.
        end_date (str|None): 'YYYY-MM-DD'. If provided with start_date, filters IC.
        reducer (str|ee.Reducer|None): Required if any feature is polygon/multipolygon.
                                       Ignored for point features.
    """

    def __init__(
        self,
        image_collection: ee.ImageCollection,
        sample_gdf: gpd.GeoDataFrame,
        identifier: str,
        out_dir: str = ".",
        selectors: Optional[List[str]] = None,
        scale: Optional[int] = None,
        crs: str = "EPSG:4326",
        crsTransform: Optional[List[float]] = None,
        num_processes: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        reducer: Optional[Union[str, ee.Reducer]] = None,
    ):
        # Filter/select up front so workers get a slim IC
        ic = image_collection
        if start_date and end_date:
            ic = ic.filterDate(start_date, end_date)
        if selectors:
            ic = ic.select(selectors)
        else:
            selectors = ic.first().bandNames().getInfo()

        self.image_collection = ic
        self.samples_gdf = sample_gdf
        self.identifier = identifier
        self.out_dir = out_dir
        self.selectors = selectors
        self.scale = scale
        self.crs = crs
        self.crsTransform = crsTransform
        self.num_processes = num_processes
        self.start_date = start_date
        self.end_date = end_date
        self.reducer = reducer

        self._validate_inputs()
        os.makedirs(self.out_dir, exist_ok=True)
        logging.basicConfig()

        # Cache features as plain JSON tuples (id, props, geom) for Pool
        self.sample_features = [
            (f["id"], f["properties"], f["geometry"])
            for f in json.loads(self.samples_gdf.to_json())["features"]
        ]

    def _validate_inputs(self):
        if not isinstance(self.image_collection, ee.ImageCollection):
            raise ValueError("image_collection must be ee.ImageCollection.")
        if self.identifier not in self.samples_gdf.columns:
            raise ValueError(
                f"Identifier column '{self.identifier}' not found in sample_gdf."
            )

        # Geometry checks and reducer requirement for polygons
        geom_types = set(self.samples_gdf.geometry.geom_type.str.upper().unique())
        allowed = {"POINT", "POLYGON", "MULTIPOLYGON"}
        if not geom_types.issubset(allowed):
            raise ValueError(
                f"Only POINT/POLYGON/MULTIPOLYGON are supported; found: {geom_types}"
            )

        has_poly = any(g in geom_types for g in ("POLYGON", "MULTIPOLYGON"))
        if has_poly:
            if self.reducer is None:
                raise ValueError(
                    "Reducer is required when sample_gdf contains polygons."
                )

            self.REDUCERS = {
                "COUNT": ee.Reducer.count(),
                "MEAN": ee.Reducer.mean(),
                "MEAN_UNWEIGHTED": ee.Reducer.mean().unweighted(),
                "MAXIMUM": ee.Reducer.max(),
                "MEDIAN": ee.Reducer.median(),
                "MINIMUM": ee.Reducer.min(),
                "MODE": ee.Reducer.mode(),
                "STD": ee.Reducer.stdDev(),
                "MIN_MAX": ee.Reducer.minMax(),
                "SUM": ee.Reducer.sum(),
                "VARIANCE": ee.Reducer.variance(),
            }

            # Normalize reducer to ee.Reducer
            if isinstance(self.reducer, str):
                key = self.reducer.upper()
                if key not in self.REDUCERS:
                    raise ValueError(
                        f"Reducer '{self.reducer}' not supported. "
                        f"Choose from: {list(self.REDUCERS.keys())}"
                    )
                self.reducer = self.REDUCERS[key]
            elif not isinstance(self.reducer, ee.Reducer):
                raise ValueError("reducer must be a string or an ee.Reducer instance.")

    def extract_timeseries(self):
        """Parallel per-feature CSV download."""

        with multiprocessing.Pool(processes=self.num_processes) as pool:
            pool.starmap(self._download_timeseries, self.sample_features)

    @retry(tries=10, delay=1, backoff=2)
    def _download_timeseries(self, id_: Union[str, int], props: dict, geom: dict):
        index_val = props[self.identifier]

        gtype = geom.get("type", "").upper()
        if gtype == "POINT":
            fc = self._fc_from_point(geom, index_val)
        elif gtype in ("POLYGON", "MULTIPOLYGON"):
            fc = self._fc_from_polygon(geom, index_val, self.reducer)
        else:
            raise ValueError(f"Unsupported geometry type: {gtype}")

        params = {"filename": f"{index_val}"}

        sels = list(self.selectors)
        if "time" not in [s.lower() for s in sels]:
            sels = ["time"] + sels
        if self.identifier not in sels:
            sels = [self.identifier] + sels
        params["selectors"] = sels

        url = fc.getDownloadURL(**params)
        resp = requests.get(url, stream=True)
        resp.raise_for_status()

        out_path = Path(self.out_dir) / f"{index_val}.csv"
        resp.encoding = resp.encoding or "utf-8"
        with open(out_path, "w", encoding=resp.encoding, newline="") as f:
            for chunk in resp.iter_content(chunk_size=65536, decode_unicode=True):
                if chunk:
                    f.write(chunk)

        print(f"Saved: {out_path}")

    def _fc_from_point(
        self, geom: dict, index_val: Union[str, int]
    ) -> ee.FeatureCollection:
        """Convert a getRegion result at a point into a FeatureCollection with 'time' as ISO and identifier."""
        coords = ee.Geometry.Point(geom["coordinates"])
        result = self.image_collection.getRegion(
            geometry=coords,
            scale=self.scale,
            crs=self.crs,
            crsTransform=self.crsTransform,
        )

        headers = result.get(0)
        rows = result.slice(1)

        def make_feature(row):
            row = ee.List(row)
            d = ee.Dictionary.fromLists(headers, row)
            date_str = ee.Date(ee.Number(d.get("time"))).format("YYYY-MM-dd HH:mm:ss")
            d = d.set("time", date_str)
            d = d.set(self.identifier, index_val)
            return ee.Feature(None, d)

        return ee.FeatureCollection(rows.map(make_feature))

    def _fc_from_polygon(
        self, geom: dict, index_val: Union[str, int], reducer: ee.Reducer
    ) -> ee.FeatureCollection:
        """Map reduceRegion over images for a polygon/multipolygon, producing one feature per image."""
        polygon = ee.Geometry(geom)
        ic = self.image_collection.filterBounds(polygon)

        def per_image(image):
            stats = image.reduceRegion(
                reducer=reducer,
                geometry=polygon,
                scale=self.scale,
                crs=self.crs,
                maxPixels=1e13,
            )
            t = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd HH:mm:ss")
            feat = ee.Feature(None, stats).set("time", t)
            feat = feat.set(self.identifier, index_val)
            return feat

        fc = ee.FeatureCollection(ic.map(per_image))
        fc = fc.filter(ee.Filter.notNull(self.image_collection.first().bandNames()))
        return fc


class ImagePatchExtractor:
    """
    Extracts image patches (chips) around sample points from an Earth Engine image.

    Args:
        image (ee.Image): Earth Engine image to extract patches from.
        samples_gdf (gpd.GeoDataFrame): GeoDataFrame of sample points with a unique identifier column.
        identifier (str): Column name in samples_gdf to use for naming patches.
        out_dir (str): Directory to save extracted patches.
        buffer (int): Buffer radius (in meters) around each point to define patch area.
        dimensions (str): Patch dimensions in the form "widthxheight", e.g., "256x256".
        format (str): Output format (e.g., "png", "jpg", "GEO_TIFF").
        num_processes (int): Number of parallel download processes.
    """

    SUPPORTED_FORMATS = {"png", "jpg", "GEO_TIFF", "ZIPPED_GEO_TIFF", "NPY"}

    def __init__(
        self,
        image: ee.Image,
        sample_gdf: gpd.GeoDataFrame,
        identifier: str,
        out_dir: str = ".",
        buffer: int = 1270,
        dimensions: str = "256x256",
        format: str = "png",
        num_processes: int = 10,
    ):
        self.image = image
        self.samples_gdf = sample_gdf
        self.identifier = identifier
        self.out_dir = out_dir
        self.buffer = buffer
        self.dimensions = dimensions
        self.format = format.upper()
        self.num_processes = num_processes

        self._validate_inputs()
        os.makedirs(self.out_dir, exist_ok=True)
        logging.basicConfig()

        self.sample_features = json.loads(self.samples_gdf.to_json())["features"]

    def _validate_inputs(self):
        # Validate dimensions format
        if not isinstance(self.dimensions, str) or "x" not in self.dimensions:
            raise ValueError(
                "dimensions must be a string in the form 'WIDTHxHEIGHT', e.g., '256x256'."
            )

        dims = self.dimensions.lower().split("x")
        if len(dims) != 2 or not all(d.isdigit() for d in dims):
            raise ValueError(
                "dimensions must contain two integers separated by 'x', e.g., '256x256'."
            )

        # Validate image format
        if self.format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: '{self.format}'. Supported formats: {self.SUPPORTED_FORMATS}"
            )

        # Validate identifier exists
        if self.identifier not in self.samples_gdf.columns:
            raise ValueError(
                f"Identifier column '{self.identifier}' not found in sample_gdf."
            )

    def extract_patches(self):
        """
        Initiates the parallel download of patches based on sample points.
        """
        items = [
            (f["id"], f["properties"], f["geometry"]) for f in self.sample_features
        ]

        pool = multiprocessing.Pool(self.num_processes)
        pool.starmap(self._download_patch, items)
        pool.close()
        pool.join()

    @retry(tries=10, delay=1, backoff=2)
    def _download_patch(self, id: Union[str, int], props: dict, geom: dict):
        """
        Downloads a single patch based on a point geometry.

        Args:
            id (str|int): Internal ID.
            props (dict): Properties from the GeoDataFrame row.
            geom (dict): Geometry of the point in GeoJSON format.
        """
        index = props[self.identifier]
        coords = ee.Geometry.Point(geom["coordinates"])
        region = coords.buffer(self.buffer).bounds()

        # Get the correct download URL based on format
        if self.format in ["PNG", "JPG"]:
            url = self.image.getThumbURL(
                {
                    "region": region,
                    "dimensions": self.dimensions,
                    "format": self.format.lower(),
                }
            )
        else:
            url = self.image.getDownloadURL(
                {"region": region, "dimensions": self.dimensions, "format": self.format}
            )

        # Determine extension
        ext = (
            "tif"
            if self.format in ["GEO_TIFF", "ZIPPED_GEO_TIFF"]
            else self.format.lower()
        )
        filename = f"{index}.{ext}"
        filepath = os.path.join(self.out_dir, filename)

        # Download and save image
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            response.raise_for_status()

        with open(filepath, "wb") as out_file:
            shutil.copyfileobj(response.raw, out_file)

        print(f"Saved: {filepath}")
