"""Module for timeseries analysis on Earth Engine data."""

import math
import ee


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
