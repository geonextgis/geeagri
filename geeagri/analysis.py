"""Module for analyzing data in Google Earth Engine."""

import math
import ee
import numpy as np
import pandas as pd
from .preprocessing import MeanCentering


class PCA:
    """
    Performs Principal Component Analysis on an Earth Engine image.

    Args:
        image (ee.Image): Multi-band image to apply PCA to.
        region (ee.Geometry): Geometry to use for statistical analysis.
        scale (int, optional): Pixel resolution for calculations. Defaults to 100.
        max_pixels (int, optional): Max number of pixels to process. Defaults to 1e9.

    Raises:
        ValueError: If input image or region is invalid.
    """

    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        max_pixels: int = int(1e9),
    ):
        if not isinstance(image, ee.Image):
            raise ValueError("`image` must be an instance of ee.Image.")
        if not isinstance(region, ee.Geometry):
            raise ValueError("`region` must be an instance of ee.Geometry.")

        self.image = image
        self.region = region
        self.scale = scale
        self._max_pixels = max_pixels

        self._scaler = MeanCentering(self.image, self.region, self.scale)
        self.centered_image = self._scaler.transform()

        self._eigen_values = None  # For storing eigenvalues for variance computation
        self._pc_names = None  # Names of the principal components

    def get_principal_components(self) -> ee.Image:
        """Computes normalized principal components of the image.

        Returns:
            ee.Image: Image with bands ['pc1', 'pc2', ..., 'pcN'] representing normalized PCs.
        """
        arrays = self.centered_image.toArray()

        covar = arrays.reduceRegion(
            reducer=ee.Reducer.centeredCovariance(),
            geometry=self.region,
            scale=self.scale,
            maxPixels=self._max_pixels,
        )

        if covar is None or covar.get("array") is None:
            raise RuntimeError(
                "Covariance matrix could not be computed. Check region/image coverage."
            )

        covar_array = ee.Array(covar.get("array"))
        eigens = covar_array.eigen()
        eigen_values = eigens.slice(1, 0, 1)
        eigen_vectors = eigens.slice(1, 1)

        self._eigen_values = eigen_values  # Save for explained variance calculation

        array_image = arrays.toArray(1)
        principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

        band_count = self.image.bandNames().size()
        band_names = ee.List.sequence(1, band_count).map(
            lambda i: ee.String("pc").cat(ee.Number(i).toInt().format())
        )
        self._pc_names = band_names

        sd_image = (
            ee.Image(eigen_values.sqrt()).arrayProject([0]).arrayFlatten([band_names])
        )

        pc_image = (
            principal_components.arrayProject([0])
            .arrayFlatten([band_names])
            .divide(sd_image)
        )

        return pc_image

    def get_explained_variance(self) -> pd.DataFrame:
        """Returns explained variance ratio for each principal component.

        Returns:
            pd.DataFrame: DataFrame with columns ['pc', 'variance_explained'].
        """
        if self._eigen_values is None:
            raise RuntimeError(
                "Call `get_principal_components()` before computing explained variance."
            )

        eigen_values = np.array(self._eigen_values.getInfo()).flatten()
        total_variance = eigen_values.sum()
        explained_variance = eigen_values / total_variance

        return pd.DataFrame(
            {"pc": self._pc_names.getInfo(), "variance_explained": explained_variance}
        )


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

        # Precompute mean composite of the selected band
        self.composite = self.image_collection.mean()

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
        self, harmonic_coeffs, cos_band, sin_band, hsv=True, stretch_factor=5
    ):
        """
        Compute phase and amplitude from harmonic coefficients. Optionally return an
        HSV-based RGB visualization.

        Args:
            harmonic_coeffs (ee.Image): Coefficients image from get_harmonic_coeffs().
            cos_band (str): Name of cosine coefficient band (e.g., '<band>_cos_1').
            sin_band (str): Name of sine coefficient band (e.g., '<band>_sin_1').
            hsv (bool, optional): If True (default), return an RGB image built from HSV
                encoding (phase → hue, amplitude → saturation, composite → value).
                If False, return the raw 'phase' (radians) and 'amplitude' bands.
            stretch_factor (float, optional): Multiplier applied to amplitude before
                mapping it to saturation in the HSV visualization.

        Returns:
            ee.Image:
                - If hsv=True: 3-band RGB image derived from HSV.
                - If hsv=False: Image with 'phase' (radians) and 'amplitude' bands.
        """

        scale = 10000

        # De-scale to original floating values
        cos = harmonic_coeffs.select(cos_band).divide(scale).toFloat()
        sin = harmonic_coeffs.select(sin_band).divide(scale).toFloat()

        # Phase in [-pi, pi], Amplitude >= 0
        phase = sin.atan2(cos).rename("phase")
        amplitude = sin.hypot(cos).rename("amplitude")

        if hsv:
            # Normalize to HSV ranges
            hsv = (
                phase.unitScale(-math.pi, math.pi)  # hue
                .addBands(amplitude.multiply(stretch_factor).clamp(0, 1))  # sat
                .addBands(self.composite)  # val
                .rename(["phase", "amplitude", "value"])
            )

            return hsv.hsvToRgb()

        else:
            return phase.addBands(amplitude)

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
