"""Module for phenology mapping in Google Earth Engine."""

import ee


class SavitzkyGolayEE:
    """Savitzky–Golay-style local polynomial fit for an Earth Engine ImageCollection.

    Performs unweighted local polynomial regression of order ``polyorder`` within a
    sliding window of ``window_length`` observations centered on each timestamp of
    the input series and returns per-timestamp regression coefficients
    ``a0..aP``. The intercept (``a0``) equals the fitted value at the window
    center—i.e., the SG-smoothed value.

    This implementation generalizes the classic Savitzky–Golay filter to the
    irregular sampling common in GEE by operating on contiguous index-based
    windows and explicitly centering time differences on the current timestamp.

    Args:
        image_collection (ee.ImageCollection): Time series to smooth. Each
            image must carry ``'system:time_start'``. If multiple bands are
            present, use ``band_name`` to choose the target band.
        window_length (int): Size of the moving window (number of
            observations). Must be an **odd** integer and at least
            ``polyorder + 1``.
        polyorder (int): Polynomial order (non-negative integer).
        band_name (str, optional): Name of the band to smooth. If omitted,
            the first band of the collection is used.

    Raises:
        ValueError: If ``image_collection`` is not an ``ee.ImageCollection``.
        ValueError: If ``window_length`` or ``polyorder`` are not integers.
        ValueError: If ``window_length`` is even.
        ValueError: If ``window_length < polyorder + 1``.
    """

    def __init__(self, image_collection, window_length, polyorder, band_name=None):
        if not isinstance(image_collection, ee.ImageCollection):
            raise ValueError("`image_collection` must be an ee.ImageCollection.")
        if not isinstance(window_length, int) or not isinstance(polyorder, int):
            raise ValueError("`window_length` and `polyorder` must be integers.")
        if window_length % 2 == 0:
            raise ValueError("`window_length` must be an odd number.")
        if window_length < (polyorder + 1):
            raise ValueError("`window_length` must be at least `polyorder + 1`.")

        self.window_length = window_length
        self.polyorder = polyorder
        self._numX = polyorder + 1
        self.half = window_length // 2

        if band_name is None:
            self._ic = image_collection.select([0], ["y"]).sort("system:time_start")
        else:
            self._ic = image_collection.select([band_name], ["y"]).sort(
                "system:time_start"
            )

        self._n = self._ic.size()
        self._img_list = self._ic.toList(self._n)
        self._coeff_names = [f"a{k}" for k in range(self._numX)]

    def _window_collection(self, idx):
        """Build the index-based window centered at a given position.

        Slices ``self._img_list`` to form a window of up to ``window_length``
        images centered on ``idx`` (clamped at the series boundaries).

        Args:
            idx (ee.Number): Center index of the window within the series.

        Returns:
            ee.ImageCollection: Contiguous index-based window around ``idx``.

        Notes:
            Using index slices avoids equality tests on floating timestamps and
            works for irregular sampling. The window will be smaller than
            ``window_length`` near the start/end of the series.
        """
        start = ee.Number(idx).subtract(self.half).max(0)
        end = ee.Number(idx).add(self.half).add(1).min(self._n)  # exclusive
        return ee.ImageCollection(self._img_list.slice(start, end))

    def _add_poly_predictors(self, window_col, center_time):
        """Add centered polynomial predictors to each image in the window.

        Creates bands ``x0..xP`` where ``x0 = 1`` and ``xk = dt^k`` for
        ``k = 1..polyorder``, with ``dt`` measured in **days** relative to
        ``center_time``. Appends the response band ``'y'`` last.

        Args:
            window_col (ee.ImageCollection): Window collection for the current fit.
            center_time (ee.Date): Timestamp at the center of the window.

        Returns:
            ee.ImageCollection: Window with predictors ``x0..xP`` followed by ``y``.

        Notes:
            The band order is important: predictors first, then the response
            ``'y'``, matching the input expected by
            ``ee.Reducer.linearRegression(numX=self._numX, numY=1)``.
        """
        numX = self._numX

        def add_poly(img):
            img = ee.Image(img)
            dt_days = ee.Number(img.date().difference(center_time, "day"))
            dt = ee.Image.constant(dt_days)
            predictors = [ee.Image.constant(1).rename("x0")] + [
                dt.pow(k).rename(f"x{k}") for k in range(1, numX)
            ]
            return ee.Image.cat(predictors + [img.select("y")]).copyProperties(
                img, ["system:time_start"]
            )

        return window_col.map(add_poly).select([*(f"x{k}" for k in range(numX)), "y"])

    def coefficients(self):
        """Compute polynomial coefficients for each timestamp.

        For every image in the input series, forms an index-based window,
        builds centered polynomial predictors, and runs a per-pixel linear
        regression over the window to obtain coefficients ``a0..aP``. Windows
        with fewer than ``polyorder + 1`` samples return **masked** outputs.

        Returns:
            ee.ImageCollection: One coefficient image per timestamp with bands
            ``a0..aP``. The image property ``'system:time_start'`` is set to the
            center timestamp.

        Examples:
            >>> coeff_ic = SavitzkyGolayEE(ic,  nine, 2, 'NDVI').coefficients()
            >>> smoothed = coeff_ic.select('a0').rename('NDVI_sg')

        See Also:
            fitted: Convenience method to return only the fitted series (``a0``).
        """
        numX = self._numX
        coeff_names = self._coeff_names

        def coeffs_at_index(idx):
            idx = ee.Number(idx)

            center = ee.Image(self._img_list.get(idx))
            center_time = ee.Date(center.get("system:time_start"))

            window_col = self._window_collection(idx)
            with_bands = self._add_poly_predictors(window_col, center_time)
            count = with_bands.size()

            fit = with_bands.reduce(ee.Reducer.linearRegression(numX=numX, numY=1))

            coeff = (
                fit.select("coefficients").arrayProject([0]).arrayFlatten([coeff_names])
            )

            result = coeff.set("system:time_start", center_time.millis())

            empty = ee.Image.cat(
                [
                    ee.Image.constant(0).updateMask(0).rename(name)
                    for name in coeff_names
                ]
            ).set("system:time_start", center_time.millis())

            return ee.Image(ee.Algorithms.If(count.gte(numX), result, empty))

        indices = ee.List.sequence(self.half, self._n.subtract(self.half + 1))
        return ee.ImageCollection(indices.map(coeffs_at_index))


class Phenometrics:
    """Slope-based phenology metrics from a coefficients ImageCollection.

    This class consumes an Earth Engine ImageCollection of local polynomial
    **coefficients** (e.g., from a Savitzky–Golay fit) that contains at least:
    - ``a0``: fitted value at each timestamp (smoothed series)
    - ``a1``: first derivative (slope) at each timestamp

    It computes, over the full time span of the collection, the following
    phenometrics using a slope-based approach:

    - **SOS** (Start of Season): pre-POS date with maximum positive slope
      (and slope ≥ ``slope_min``).
    - **POS** (Peak of Season): date of maximum fitted value (``a0``).
    - **EOS** (End of Season): post-POS date with most negative slope
      (and slope ≤ −``slope_min``).
    - **LOS** (Length of Season): EOS − SOS (days).
    - **IOS** (Integrated greenness): trapezoidal integral of ``a0`` between
      SOS and EOS (units: fitted_units·days, e.g., NDVI·days).

    The output image includes:
    - ``SOS_DOY``, ``POS_DOY``, ``EOS_DOY`` (calendar day-of-year, 1..366)
    - ``LOS_days`` (days)
    - ``IOS`` (value·days)
    - ``POS_value`` (fitted value at POS)
    - ``AMP`` (seasonal amplitude = POS − MIN within window)
    - ``SOS_value`` and ``EOS_value`` (fitted values at SOS/EOS)

    Args:
        image_collection (ee.ImageCollection):
            Coefficients collection with bands ``a0`` and ``a1`` and the
            property ``system:time_start`` on each image.
        slope_min (float, optional):
            Minimum absolute slope (per day) required when selecting SOS/EOS
            candidates. Defaults to 0.002.
        min_amp (float, optional):
            Minimum amplitude (same units as ``a0``) required to keep the
            season. Defaults to 0.10.

    Notes:
        * The analysis window spans the full extent of the input collection
          (from its earliest to its latest timestamp, inclusive).
        * DOY is the **calendar** day-of-year computed from each image date
          (Jan 1 → 1).
    """

    def __init__(self, image_collection, slope_min=0.002, min_amp=0.10):
        if not isinstance(image_collection, ee.ImageCollection):
            raise ValueError("image_collection must be an ee.ImageCollection.")

        self.slope_min = float(slope_min)
        self.min_amp = float(min_amp)

        self._ic_raw = image_collection.sort("system:time_start")
        self._day_ms = 86400000.0

        # Window (inclusive): earliest .. latest timestamp present
        first_img = ee.Image(self._ic_raw.first())
        last_img = ee.Image(self._ic_raw.sort("system:time_start", False).first())
        self._start = ee.Date(first_img.get("system:time_start"))
        self._end = ee.Date(last_img.get("system:time_start"))
        end_inclusive = self._end.advance(1, "second")

        # Keep only a0,a1 and add helper bands:
        #   y  = a0 (fitted value)
        #   d1 = a1 (slope per day)
        #   t  = time in days since epoch (float)
        #   doy = calendar day-of-year (1..366, int16)
        def _add_time_bands(img):
            img = ee.Image(img)
            t_days = ee.Number(img.get("system:time_start")).divide(self._day_ms)
            doy = img.date().getRelative("day", "year").add(1)  # 1..366
            return (
                img.select(["a0", "a1"], ["y", "d1"])
                .addBands(ee.Image.constant(t_days).rename("t").toFloat())
                .addBands(ee.Image.constant(doy).rename("doy").toInt16())
                .copyProperties(img, ["system:time_start"])
            )

        self._ic = (
            self._ic_raw.filterDate(self._start, end_inclusive)
            .map(_add_time_bands)
            .sort("system:time_start")
        )

        self._n = self._ic.size()

    def _pos_and_amplitude(self):
        """Return (t_pos_days, y_pos, doy_pos, y_min, amp, amp_mask)."""
        # Peak (POS): maximize y
        pos_img = self._ic.qualityMosaic("y")
        t_pos_days = pos_img.select("t")
        y_pos = pos_img.select("y")
        doy_pos = pos_img.select("doy")

        # Minimum fitted value in window (negative trick)
        min_ic = self._ic.map(
            lambda im: ee.Image(im).addBands(
                ee.Image(im).select("y").multiply(-1).rename("neg")
            )
        )
        min_img = min_ic.qualityMosaic("neg")
        y_min = min_img.select("y")

        amp = y_pos.subtract(y_min)
        amp_mask = amp.gte(self.min_amp)
        return t_pos_days, y_pos, doy_pos, amp, amp_mask

    def _sos_eos(self, t_pos_days):
        """Find SOS/EOS timestamps (and their fitted values)."""
        # SOS: pre-POS, maximize d1 subject to d1 >= slope_min
        d1_pre = self._ic.map(
            lambda im: ee.Image(im)
            .updateMask(im.select("t").lt(t_pos_days))
            .updateMask(im.select("d1").gte(self.slope_min))
        )
        sos_img = d1_pre.qualityMosaic("d1")
        t_sos_days = sos_img.select("t")
        doy_sos = sos_img.select("doy")
        y_sos = sos_img.select("y")

        # EOS: post-POS, minimize d1 (maximize -d1) with d1 <= -slope_min
        d1neg_post = self._ic.map(
            lambda im: ee.Image(im)
            .addBands(im.select("d1").multiply(-1).rename("d1neg"))
            .updateMask(im.select("t").gt(t_pos_days))
            .updateMask(im.select("d1").lte(-self.slope_min))
            .updateMask(im.select("y").lte(y_sos))
        )
        eos_img = d1neg_post.qualityMosaic("d1neg")
        t_eos_days = eos_img.select("t")
        doy_eos = eos_img.select("doy")
        y_eos = eos_img.select("y")

        return t_sos_days, doy_sos, y_sos, t_eos_days, doy_eos, y_eos

    def _ios(self, t_sos_days, t_eos_days):
        """Trapezoidal integral of y between SOS and EOS (value·days)."""
        n = self._n
        y_list = self._ic.toList(n)

        def _seg_area_at(i):
            i = ee.Number(i)
            prev = ee.Image(y_list.get(i.subtract(1)))
            curr = ee.Image(y_list.get(i))
            t_prev = prev.select("t")
            t_curr = curr.select("t")
            y_prev = prev.select("y")
            y_curr = curr.select("y")

            dt = t_curr.subtract(t_prev)  # days
            area = y_prev.add(y_curr).divide(2.0).multiply(dt).rename("seg_area")
            t_mid = t_prev.add(t_curr).divide(2.0)
            return area.updateMask(
                t_mid.gte(t_sos_days).And(t_mid.lte(t_eos_days))
            ).set("system:time_start", prev.get("system:time_start"))

        seg_images = ee.Algorithms.If(
            n.lte(1), ee.List([]), ee.List.sequence(1, n.subtract(1)).map(_seg_area_at)
        )
        return ee.ImageCollection.fromImages(seg_images).sum().rename("IOS")

    def metrics(self):
        """Compute SOS, POS, EOS, LOS, IOS, POS_value, AMP (and SOS/EOS values).

        Returns:
            ee.Image: Bands
                - SOS (int16): start of season (calendar DOY, 1..366)
                - POS (int16): peak of season (calendar DOY)
                - EOS (int16): end of season (calendar DOY)
                - LOS (int16): length of season (days)
                - IOS (float): integral of fitted value between SOS and EOS
                - vPOS (float): fitted value at POS
                - AMP (float): amplitude = POS − MIN within window
                - vSOS (float): fitted value at SOS
                - vEOS (float): fitted value at EOS
        """
        t_pos_days, y_pos, doy_pos, amp, amp_mask = self._pos_and_amplitude()
        t_sos_days, doy_sos, y_sos, t_eos_days, doy_eos, y_eos = self._sos_eos(
            t_pos_days
        )
        ios = self._ios(t_sos_days, t_eos_days).toFloat()

        los_days = t_eos_days.subtract(t_sos_days).rename("LOS").toInt16()

        out = ee.Image.cat(
            [
                doy_sos.rename("SOS").toInt16(),
                doy_pos.rename("POS").toInt16(),
                doy_eos.rename("EOS").toInt16(),
                los_days,
                ios.rename("IOS"),
                y_pos.rename("vPOS"),
                amp.rename("AMP"),
                y_sos.rename("vSOS"),
                y_eos.rename("vEOS"),
            ]
        ).updateMask(amp_mask)

        return out.set(
            {
                "method": "slope",
                "window_start": self._start.format("YYYY-MM-dd"),
                "window_end": self._end.format("YYYY-MM-dd"),
            }
        )
