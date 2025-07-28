"""Module for analyzing data in Google Earth Engine."""

import ee
from .preprocessing import MeanCentering


class PCA:
    def __init__(
        self,
        image: ee.Image,
        region: ee.Geometry,
        scale: int = 100,
        max_pixels: int = int(1e9),
    ):
        self.image = image
        self.region = region
        self.scale = scale
        self._max_pixels = max_pixels
        self._scaler = MeanCentering(self.image, self.region, self.scale)
        self.centered_image = self._scaler.transform()

    def get_principal_components(self):
        # Collapse bands into 1D array
        arrays = self.centered_image.toArray()

        # Compute the covariance of the bands within the region.
        covar = arrays.reduceRegion(
            reducer=ee.Reducer.centeredCovariance(),
            geometry=self.region,
            scale=self.scale,
            maxPixels=self.max_pixels,
        )

        # Get the 'array' covariance result and cast to an array.
        # This represents the band-to-band covariance within the region.
        covar_array = ee.Array(covar.get("array"))

        # Perform an eigen analysis and slice apart the values and vectors.
        eigen = covar_array.eigen()

        # This is a P-length vector of Eigenvalues.
        eigen_values = eigens.slice(1, 0, 1)  # (axis, start, end)
        # This is a PxP matrix with eigenvectors in rows.
        eigen_vectors = eigens.slice(1, 1)  # (axis, start)

        # Convert the array image to 2D arrays for matrix computations.
        array_image = arrays.toArray(1)

        # Left multiply the image array by the matrix of eigenvectors.
        principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

        # Turn the square roots of the Eigenvalues into a P-band image.
        sd_image = (
            ee.Image(eigen_values.sqrt())
            .arrayProject([0])
            .arrayFlatten([get_new_band_names("sd")])
        )

        # Turn the PCs into a P-band image, normalized by SD.
        return (
            # Throw out an an unneeded dimension, [[]] -> [].
            principal_components.arrayProject([0])
            # Make the one band array image a multi-band image, [] -> image.
            .arrayFlatten([get_new_band_names("pc")])
            # Normalize the PCs by their SDs.
            .divide(sd_image)
        )
