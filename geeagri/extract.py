"""Module for extracting data from Google Earth Engine."""

import math
import ee


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
    import pandas as pd
    from datetime import datetime

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
    import pandas as pd
    from datetime import datetime

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
