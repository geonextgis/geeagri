"""Module for preprocessing Earth Observation data using Google Earth Engine."""

import ee


class Sentinel2CloudMask:
    """A utility class for creating cloud- and shadow-masked Sentinel-2 image collections.

    This class uses Sentinel-2 Level-2A Surface Reflectance (SR) data in combination
    with Sentinel-2 Cloud Probability (s2cloudless) data to generate a
    cloud-free ImageCollection.

    Attributes:
        region (ee.Geometry): The region of interest for filtering the ImageCollection.
        start_date (str): Start date (inclusive) in 'YYYY-MM-DD' format.
        end_date (str): End date (exclusive) in 'YYYY-MM-DD' format.
        cloud_filter (int): Maximum scene-level cloudiness allowed (%).
        cloud_prob_threshold (int): Cloud probability threshold (values above are considered clouds).
        nir_dark_threshold (float): NIR reflectance threshold (values below considered potential shadows).
        shadow_proj_dist (int): Maximum distance (km) to search for shadows from clouds.
        buffer (int): Buffer distance (m) to dilate cloud/shadow masks.
    """

    def __init__(
        self,
        region,
        start_date,
        end_date,
        cloud_filter=60,
        cloud_prob_threshold=50,
        nir_dark_threshold=0.15,
        shadow_proj_dist=1,
        buffer=50,
    ):

        if not isinstance(region, ee.Geometry):
            raise ValueError("`region` must be an instance of ee.Geometry.")

        self.region = region
        self.start_date = start_date
        self.end_date = end_date
        self.cloud_filter = cloud_filter
        self.cloud_prob_threshold = cloud_prob_threshold
        self.nir_dark_threshold = nir_dark_threshold
        self.shadow_proj_dist = shadow_proj_dist
        self.buffer = buffer

    def get_cloud_collection(self):
        """Retrieve Sentinel-2 images joined with s2cloudless cloud probability.

        Returns:
            ee.ImageCollection: Sentinel-2 SR images with a property containing
            the matching s2cloudless image.
        """
        s2_sr = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", self.cloud_filter))
        )

        s2_cloud_prob = (
            ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
            .filterBounds(self.region)
            .filterDate(self.start_date, self.end_date)
        )

        joined = ee.ImageCollection(
            ee.Join.saveFirst("cloud_prob").apply(
                primary=s2_sr,
                secondary=s2_cloud_prob,
                condition=ee.Filter.equals(
                    leftField="system:index", rightField="system:index"
                ),
            )
        )

        return joined

    def _add_cloud_bands(self, image):
        """Add cloud probability and binary cloud mask bands.

        Args:
            image (ee.Image): Sentinel-2 image.

        Returns:
            ee.Image: Image with added `cloud_prob` and `clouds` bands.
        """
        cloud_prob = ee.Image(image.get("cloud_prob")).select("probability")
        is_cloud = cloud_prob.gt(self.cloud_prob_threshold).rename("clouds")

        return image.addBands([cloud_prob.rename("cloud_prob"), is_cloud])

    def _add_shadow_bands(self, image):
        """Add potential shadow bands to the image.

        Args:
            image (ee.Image): Sentinel-2 image with cloud mask.

        Returns:
            ee.Image: Image with added `dark_pixels`, `cloud_transform`, and `shadows` bands.
        """
        not_water = image.select("SCL").neq(6)

        scale_factor = 1e4
        dark_pixels = (
            image.select("B8")
            .lt(self.nir_dark_threshold * scale_factor)
            .multiply(not_water)
            .rename("dark_pixels")
        )

        shadow_azimuth = ee.Number(90).subtract(
            ee.Number(image.get("MEAN_SOLAR_AZIMUTH_ANGLE"))
        )

        cloud_proj = (
            image.select("clouds")
            .directionalDistanceTransform(shadow_azimuth, self.shadow_proj_dist * 10)
            .reproject(crs=image.select(0).projection(), scale=100)
            .select("distance")
            .mask()
            .rename("cloud_transform")
        )

        shadows = cloud_proj.multiply(dark_pixels).rename("shadows")

        return image.addBands([dark_pixels, cloud_proj, shadows])

    def _add_cloud_shadow_mask(self, image):
        """Create combined cloud + shadow mask.

        Args:
            image (ee.Image): Sentinel-2 image.

        Returns:
            ee.Image: Image with an added `cloudmask` band.
        """
        image = self._add_cloud_bands(image)
        image = self._add_shadow_bands(image)

        cloud_shadow_mask = image.select("clouds").add(image.select("shadows")).gt(0)

        cloud_shadow_mask = (
            cloud_shadow_mask.focal_min(2)
            .focal_max(self.buffer * 2 / 20)
            .reproject(crs=image.select(0).projection(), scale=20)
            .rename("cloudmask")
        )

        return image.addBands(cloud_shadow_mask)

    def _apply_cloud_shadow_mask(self, image):
        """Apply cloud/shadow mask to reflectance bands.

        Args:
            image (ee.Image): Sentinel-2 image with `cloudmask` band.

        Returns:
            ee.Image: Cloud/shadow-masked image (reflectance bands only).
        """
        not_cloud_shadow = image.select("cloudmask").Not()
        return image.select("B.*").updateMask(not_cloud_shadow)

    def get_cloudfree_collection(self):
        """Generate cloud-free Sentinel-2 ImageCollection.

        Returns:
            ee.ImageCollection: Cloud- and shadow-masked Sentinel-2 SR collection.
        """
        cloud_collection = self.get_cloud_collection()
        return cloud_collection.map(self._add_cloud_shadow_mask).map(
            self._apply_cloud_shadow_mask
        )


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
