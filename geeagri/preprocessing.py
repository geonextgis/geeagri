
"""
geeagri.preprocessing
=====================

This module provides preprocessing utilities for Earth Observation data using Google Earth Engine (GEE).
It includes classes for band-wise normalization and scaling of multi-band images, enabling standardized analysis and machine learning workflows.

Available scalers:

- MeanCentering: Subtracts the mean of each band over a region.
- StandardScaler: Applies z-score normalization (zero mean, unit variance).
- MinMaxScaler: Scales each band to [0, 1] using min and max values.
- RobustScaler: Scales each band to [0, 1] using percentiles, reducing the influence of outliers.

Each scaler operates on an ee.Image and a region (ee.Geometry), and supports custom scale and pixel limits.

Example usage:
    from geeagri.preprocessing import StandardScaler
    scaler = StandardScaler(image, region)
    standardized = scaler.transform()

These tools are useful for preparing satellite imagery for further analysis, feature extraction, or machine learning tasks.
"""

import ee


class MeanCentering:
    """
    Mean-centers each band of an Earth Engine image.

    The transformation is computed as:
        X_centered = X - μ

    Where:
        X: original pixel value
        μ: mean of the band computed over the given region

    Attributes:
        image (ee.Image): The input multi-band image.
        region (ee.Geometry): The region over which to compute the mean.
        scale (int): Resolution in meters for computation.
        max_pixels (int): Maximum number of pixels for region reduction.
    """

    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        max_pixels: int = int(1e9),
    ):
        """
        Initializes the MeanCentering object.

        Args:
            image (ee.Image): Input multi-band image to center.
            region (ee.Geometry): Geometry over which statistics will be computed.
            scale (int, optional): Spatial resolution in meters. Defaults to 100.
            max_pixels (int, optional): Max pixels allowed in computation. Defaults to 1e9.

        Raises:
            TypeError: If image or region is not an ee.Image or ee.Geometry.
        """
        if not isinstance(image, ee.Image):
            raise TypeError("Expected 'image' to be of type ee.Image.")
        if not isinstance(region, ee.Geometry):
            raise TypeError("Expected 'region' to be of type ee.Geometry.")

        self.image = image
        self.region = region
        self.scale = scale
        self.max_pixels = max_pixels

    def transform(self) -> ee.Image:
        """
        Applies mean-centering to each band of the image.

        Returns:
            ee.Image: The centered image with mean of each band subtracted.

        Raises:
            ValueError: If mean computation returns None or missing values.
        """
        means = self.image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.region,
            scale=self.scale,
            bestEffort=True,
            maxPixels=self.max_pixels,
        )

        if means is None:
            raise ValueError("Mean computation failed — no valid pixels in the region.")

        bands = self.image.bandNames()

        def center_band(band):
            band = ee.String(band)
            mean = ee.Number(means.get(band))
            if mean is None:
                raise ValueError(f"Mean value not found for band: {band.getInfo()}")
            return self.image.select(band).subtract(mean).rename(band)

        centered = bands.map(center_band)
        return ee.ImageCollection(centered).toBands().rename(bands)


class StandardScaler:
    """
    Standardizes each band of an Earth Engine image using z-score normalization.

    The transformation is computed as:
        X_standardized = (X - μ) / σ

    Where:
        X: original pixel value
        μ: mean of the band over the region
        σ: standard deviation of the band over the region
    """

    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        max_pixels: int = int(1e9),
    ):
        if not isinstance(image, ee.Image):
            raise TypeError("Expected 'image' to be of type ee.Image.")
        if not isinstance(region, ee.Geometry):
            raise TypeError("Expected 'region' to be of type ee.Geometry.")

        self.image = image
        self.region = region
        self.scale = scale
        self.max_pixels = max_pixels

    def transform(self) -> ee.Image:
        """
        Applies z-score normalization to each band.

        Returns:
            ee.Image: Standardized image with zero mean and unit variance.

        Raises:
            ValueError: If statistics could not be computed.
        """
        means = self.image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=self.region,
            scale=self.scale,
            bestEffort=True,
            maxPixels=self.max_pixels,
        )
        stds = self.image.reduceRegion(
            reducer=ee.Reducer.stdDev(),
            geometry=self.region,
            scale=self.scale,
            bestEffort=True,
            maxPixels=self.max_pixels,
        )

        if means is None or stds is None:
            raise ValueError(
                "Statistic computation failed — check if region has valid pixels."
            )

        bands = self.image.bandNames()

        def scale_band(band):
            band = ee.String(band)
            mean = ee.Number(means.get(band))
            std = ee.Number(stds.get(band))
            if mean is None or std is None:
                raise ValueError(f"Missing stats for band: {band.getInfo()}")
            return self.image.select(band).subtract(mean).divide(std).rename(band)

        scaled = bands.map(scale_band)
        return ee.ImageCollection(scaled).toBands().rename(bands)


class MinMaxScaler:
    """
    Applies min-max normalization to each band of an Earth Engine image.

    The transformation is computed as:
        X_scaled = (X - min) / (max - min)
        X_scaled ∈ [0, 1] after clamping

    Where:
        min, max: band-wise min and max over the region.
    """

    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        max_pixels: int = int(1e9),
    ):
        if not isinstance(image, ee.Image):
            raise TypeError("Expected 'image' to be of type ee.Image.")
        if not isinstance(region, ee.Geometry):
            raise TypeError("Expected 'region' to be of type ee.Geometry.")

        self.image = image
        self.region = region
        self.scale = scale
        self.max_pixels = max_pixels

    def transform(self) -> ee.Image:
        """
        Applies min-max scaling to [0, 1] per band.

        Returns:
            ee.Image: Scaled image with values in range [0, 1].

        Raises:
            ValueError: If min or max statistics are missing.
        """
        stats = self.image.reduceRegion(
            reducer=ee.Reducer.minMax(),
            geometry=self.region,
            scale=self.scale,
            bestEffort=True,
            maxPixels=self.max_pixels,
        )

        if stats is None:
            raise ValueError(
                "MinMax reduction failed — possibly no valid pixels in region."
            )

        bands = self.image.bandNames()

        def scale_band(band):
            band = ee.String(band)
            min_val = ee.Number(stats.get(band.cat("_min")))
            max_val = ee.Number(stats.get(band.cat("_max")))
            if min_val is None or max_val is None:
                raise ValueError(f"Missing min/max for band: {band.getInfo()}")
            scaled = (
                self.image.select(band)
                .subtract(min_val)
                .divide(max_val.subtract(min_val))
            )
            return scaled.clamp(0, 1).rename(band)

        scaled = bands.map(scale_band)
        return ee.ImageCollection(scaled).toBands().rename(bands)


class RobustScaler:
    """
    Scales each band of an Earth Engine image using percentiles to reduce the influence of outliers.

    The transformation is computed as:
        X_scaled = (X - P_lower) / (P_upper - P_lower)
        X_scaled ∈ [0, 1] after clamping

    Where:
        - X: pixel value
        - P_lower: lower percentile value (e.g. 5th)
        - P_upper: upper percentile value (e.g. 95th)

    Attributes:
        image (ee.Image): Input multi-band image.
        region (ee.Geometry): Geometry over which percentiles are computed.
        scale (int): Resolution in meters for computation.
        lower (int): Lower percentile (default: 25).
        upper (int): Upper percentile (default: 75).
        max_pixels (int): Max pixel limit for reduceRegion.
    """

    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        lower: int = 25,
        upper: int = 75,
        max_pixels: int = int(1e9),
    ):
        """
        Initializes the RobustScaler.

        Args:
            image (ee.Image): Input image.
            region (ee.Geometry): Region to compute percentiles.
            scale (int, optional): Spatial resolution in meters. Defaults to 100.
            lower (int, optional): Lower percentile. Defaults to 5.
            upper (int, optional): Upper percentile. Defaults to 95.
            max_pixels (int, optional): Max pixels allowed for computation. Defaults to 1e9.

        Raises:
            TypeError: If image or region is of wrong type.
            ValueError: If percentiles are not in valid range.
        """
        if not isinstance(image, ee.Image):
            raise TypeError("Expected 'image' to be of type ee.Image.")
        if not isinstance(region, ee.Geometry):
            raise TypeError("Expected 'region' to be of type ee.Geometry.")
        if not (0 <= lower < upper <= 100):
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

        self.image = image
        self.region = region
        self.scale = scale
        self.lower = lower
        self.upper = upper
        self.max_pixels = max_pixels

    def transform(self) -> ee.Image:
        """
        Applies percentile-based scaling to each band in the image.
        Values are scaled to the [0, 1] range and clamped.

        Returns:
            ee.Image: The scaled image with values between 0 and 1.

        Raises:
            ValueError: If percentile reduction fails.
        """
        bands = self.image.bandNames()
        percentiles = self.image.reduceRegion(
            reducer=ee.Reducer.percentile([self.lower, self.upper]),
            geometry=self.region,
            scale=self.scale,
            bestEffort=True,
            maxPixels=self.max_pixels,
        )

        if percentiles is None:
            raise ValueError("Percentile computation failed.")

        def scale_band(band):
            band = ee.String(band)
            p_min = ee.Number(percentiles.get(band.cat(f"_p{self.lower}")))
            p_max = ee.Number(percentiles.get(band.cat(f"_p{self.upper}")))
            if p_min is None or p_max is None:
                raise ValueError(
                    f"Missing percentile values for band: {band.getInfo()}"
                )

            scaled = (
                self.image.select(band).subtract(p_min).divide(p_max.subtract(p_min))
            )
            return scaled.clamp(0, 1).rename(band)

        scaled = bands.map(scale_band)
        return ee.ImageCollection(scaled).toBands().rename(bands)
