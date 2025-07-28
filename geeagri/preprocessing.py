"""Module for preprocessing Earth Observation data using Google Earth Engine."""

import ee


class MeanCentering:
    r"""
    Mean-centers each band of an Earth Engine image.

    The transformation is computed as:

    $$
    X_{centered} = X - \mu
    $$

    Where:

    - $X$: original pixel value  
    - $\mu$: mean of the band computed over the given region
    
    Args:
        image (ee.Image): Input multi-band image to center.
        region (ee.Geometry): Geometry over which statistics will be computed.
        scale (int, optional): Spatial resolution in meters. Defaults to 100.
        max_pixels (int, optional): Max pixels allowed in computation. Defaults to 1e9.

    Raises:
        TypeError: If image or region is not an ee.Image or ee.Geometry.
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


class MinMaxScaler:
    r"""
    Applies min-max normalization to each band of an Earth Engine image.

    The transformation is computed as:

    $$
    X_\\text{scaled} = \\frac{X - \\min}{\\max - \\min}
    $$

    After clamping, $X_\\text{scaled} \\in [0, 1]$.

    Where:

    - $\min$, $\max$: band-wise minimum and maximum values over the region.

    Args:
        image (ee.Image): The input multi-band image.
        region (ee.Geometry): The region over which to compute min and max.
        scale (int, optional): The spatial resolution in meters. Defaults to 100.
        max_pixels (int, optional): Max pixels allowed during reduction. Defaults to 1e9.

    Raises:
        TypeError: If `image` is not an `ee.Image` or `region` is not an `ee.Geometry`.
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
        Applies min-max scaling to each band, producing values in the range [0, 1].

        Returns:
            ee.Image: A scaled image with band values clamped between 0 and 1.

        Raises:
            ValueError: If min or max statistics are unavailable or reduction fails.
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


class StandardScaler:
    r"""
    Standardizes each band of an Earth Engine image using z-score normalization.

    The transformation is computed as:

    $$
    X_\\text{standardized} = \\frac{X - \\mu}{\\sigma}
    $$

    Where:

    - $X$: original pixel value  
    - $\mu$: mean of the band over the specified region  
    - $\sigma$: standard deviation of the band over the specified region

    This transformation results in a standardized image where each band has
    zero mean and unit variance (approximately), assuming normally distributed values.
    
    Args:
        image (ee.Image): The input multi-band image to be standardized.
        region (ee.Geometry): The geographic region over which to compute the statistics.
        scale (int, optional): Spatial resolution (in meters) to use for region reduction. Defaults to 100.
        max_pixels (int, optional): Maximum number of pixels allowed in reduction. Defaults to 1e9.

    Raises:
        TypeError: If `image` is not an `ee.Image` or `region` is not an `ee.Geometry`.
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


class RobustScaler:
    r"""
    Applies robust scaling to each band of an Earth Engine image using percentiles,
    which reduces the influence of outliers compared to min-max scaling.

    The transformation is computed as:

    $$
    X_\\text{scaled} = \\frac{X - P_{\\text{lower}}}{P_{\\text{upper}} - P_{\\text{lower}}}
    $$

    After clamping, $X_\\text{scaled} \\in [0, 1]$.

    Where:

    - $X$: original pixel value  
    - $P_{\\text{lower}}$: lower percentile value (e.g., 25th percentile)  
    - $P_{\\text{upper}}$: upper percentile value (e.g., 75th percentile)

    This method is particularly useful when the image contains outliers or skewed distributions.

    Args:
        image (ee.Image): The input multi-band image.
        region (ee.Geometry): Geometry over which percentiles are computed.
        scale (int): Spatial resolution in meters for computation.
        lower (int): Lower percentile to use (default: 25).
        upper (int): Upper percentile to use (default: 75).
        max_pixels (int): Maximum number of pixels allowed for region reduction.
        
    Raises:
        TypeError: If `image` is not an `ee.Image` or `region` is not an `ee.Geometry`.
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
