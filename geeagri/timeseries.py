"""Module for timeseries analysis on Earth Engine data."""

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


class HarmonicRegression:
    """
    Perform harmonic regression on an Earth Engine ImageCollection.

    Attributes:
        image_collection (ee.ImageCollection): Input time series of selected band.
        ref_date (ee.Date): Reference date to calculate time.
        band (str): Name of dependent variable band.
        order (int): Number of harmonics.
        omega (float): Base frequency multiplier.
        independents (List[str]): Names of independent variable bands.
        composite (ee.Image): Median composite of the selected band.
    """

    def __init__(self, image_collection, ref_date, band_name, order=1, omega=1):
        """
        Initialize the HarmonicRegression object.

        Args:
            image_collection (ee.ImageCollection): Input image collection.
            ref_date (str or ee.Date): Reference date to compute relative time.
            band_name (str): Name of dependent variable band.
            order (int): Number of harmonics (default 1).
            omega (float): Base frequency multiplier (default 1).
        """
        self.image_collection = image_collection.select(band_name)
        self.ref_date = ee.Date(ref_date) if isinstance(ref_date, str) else ref_date
        self.band = band_name
        self.order = order
        self.omega = omega

        # Names of independent variables: constant, cos_1, ..., sin_1, ...
        self.independents = (
            ["constant"]
            + [f"cos_{i}" for i in range(1, order + 1)]
            + [f"sin_{i}" for i in range(1, order + 1)]
        )

        # Precompute median composite of the selected band
        self.composite = self.image_collection.median()

    def _add_time_unit(self, image):
        """
        Add time difference in years from ref_date as band 't'.

        Args:
            image (ee.Image): Input image.

        Returns:
            ee.Image: Image with additional 't' band.
        """
        dyear = ee.Number(image.date().difference(self.ref_date, "year"))
        return image.addBands(ee.Image.constant(dyear).rename("t").float())

    def _add_harmonics(self, image):
        """
        Add harmonic basis functions: constant, cos_i, sin_i bands.

        Args:
            image (ee.Image): Input image.

        Returns:
            ee.Image: Image with added harmonic bands.
        """
        image = self._add_time_unit(image)
        t = image.select("t")

        harmonic_bands = [ee.Image.constant(1).rename("constant")]
        for i in range(1, self.order + 1):
            freq = ee.Number(i).multiply(self.omega).multiply(2 * math.pi)
            harmonic_bands.append(t.multiply(freq).cos().rename(f"cos_{i}"))
            harmonic_bands.append(t.multiply(freq).sin().rename(f"sin_{i}"))

        return image.addBands(ee.Image(harmonic_bands))

    def get_harmonic_coeffs(self):
        """
        Fit harmonic regression and return coefficients image.

        Returns:
            ee.Image: Coefficients image with bands like <band>_constant, <band>_cos_1, etc.
        """
        harmonic_coll = self.image_collection.map(self._add_harmonics)

        regression = harmonic_coll.select(self.independents + [self.band]).reduce(
            ee.Reducer.linearRegression(len(self.independents), 1)
        )

        coeffs = (
            regression.select("coefficients")
            .arrayProject([0])
            .arrayFlatten([self.independents])
            .multiply(10000)
            .toInt32()
        )

        new_names = [f"{self.band}_{name}" for name in self.independents]
        return coeffs.rename(new_names)

    def get_phase_amplitude(
        self, harmonic_coeffs, cos_band, sin_band, stretch_factor=1, return_rgb=True
    ):
        """
        Compute phase & amplitude and optionally create RGB visualization.

        Args:
            harmonic_coeffs (ee.Image): Coefficients image from get_harmonic_coeffs().
            cos_band (str): Name of cosine coefficient band.
            sin_band (str): Name of sine coefficient band.
            stretch_factor (float): Stretch amplitude to enhance contrast.
            return_rgb (bool): If True, return RGB image; else return HSV image.

        Returns:
            ee.Image: RGB visualization (uint8) or HSV image.
        """
        phase = harmonic_coeffs.select(cos_band).atan2(harmonic_coeffs.select(sin_band))
        amplitude = harmonic_coeffs.select(cos_band).hypot(
            harmonic_coeffs.select(sin_band)
        )

        hsv = (
            phase.unitScale(-math.pi, math.pi)
            .addBands(amplitude.multiply(stretch_factor))
            .addBands(self.composite)
        )

        if return_rgb:
            return hsv.hsvToRgb().unitScale(0, 1).multiply(255).toByte()
        else:
            return hsv

    def _fit_harmonics(self, harmonic_coeffs, image):
        """
        Compute fitted values from harmonic coefficients and harmonic bands.

        Args:
            harmonic_coeffs (ee.Image): Coefficients image divided by 10000.
            image (ee.Image): Image with harmonic bands.

        Returns:
            ee.Image: Image with fitted values.
        """
        return (
            image.select(self.independents)
            .multiply(harmonic_coeffs)
            .reduce("sum")
            .rename("fitted")
            .copyProperties(image, ["system:time_start"])
        )

    def get_fitted_harmonics(self, harmonic_coeffs):
        """
        Compute fitted harmonic time series over the collection.

        Args:
            harmonic_coeffs (ee.Image): Coefficients image from get_harmonic_coeffs().

        Returns:
            ee.ImageCollection: Collection with fitted harmonic value as 'fitted' band.
        """
        harmonic_coeffs_scaled = harmonic_coeffs.divide(10000)
        harmonic_coll = self.image_collection.map(self._add_harmonics)

        return harmonic_coll.map(
            lambda img: self._fit_harmonics(harmonic_coeffs_scaled, img)
        )
