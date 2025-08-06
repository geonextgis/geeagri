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
from typing import Union


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
