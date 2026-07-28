"""Preprocessing utilities for 3D medical images.

Provides intensity normalisation, clipping, denoising (Gaussian / median /
bilateral) and N4 bias-field correction (via SimpleITK).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["Preprocessor"]


class Preprocessor:
    """Static collection of preprocessing routines.

    Every method takes (and returns) a 3D ``np.ndarray``.  All methods are
    ``@staticmethod`` — :class:`Preprocessor` is never instantiated.
    """

    # ------------------------------------------------------------------ #
    #  Intensity operations                                               #
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(volume: np.ndarray, mode: str = "minmax") -> np.ndarray:
        """Normalise intensities to ``[0, 1]`` (``minmax``) or zero-mean/unit-variance (``zscore``).

        Parameters
        ----------
        volume : np.ndarray
            3D input volume.
        mode : str, {'minmax', 'zscore'}
            Normalisation strategy.

        Returns
        -------
        np.ndarray
            Normalised float32 volume.
        """
        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return vol
        if mode == "minmax":
            lo, hi = float(vol.min()), float(vol.max())
            if hi - lo < 1e-8:
                return np.zeros_like(vol)
            return (vol - lo) / (hi - lo)
        elif mode == "zscore":
            mu, sigma = float(vol.mean()), float(vol.std())
            if sigma < 1e-8:
                return vol - mu
            return (vol - mu) / sigma
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'minmax' or 'zscore'.")

    @staticmethod
    def clip_intensity(
        volume: np.ndarray,
        p_low: float = 1.0,
        p_high: float = 99.0,
    ) -> np.ndarray:
        """Clip intensities to the ``[p_low, p_high]`` percentile range.

        Robust to outliers (e.g. metal artifacts, fat deposits).
        """
        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return vol
        lo = float(np.percentile(vol, p_low))
        hi = float(np.percentile(vol, p_high))
        return np.clip(vol, lo, hi).astype(np.float32)

    @staticmethod
    def rescale(
        volume: np.ndarray,
        out_min: float = 0.0,
        out_max: float = 255.0,
    ) -> np.ndarray:
        """Linearly rescale the volume to ``[out_min, out_max]``."""
        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return vol
        lo, hi = float(vol.min()), float(vol.max())
        if hi - lo < 1e-8:
            return np.full_like(vol, out_min, dtype=np.float32)
        return out_min + (vol - lo) * (out_max - out_min) / (hi - lo)

    # ------------------------------------------------------------------ #
    #  Denoising                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def gaussian_smooth(
        volume: np.ndarray,
        sigma: float = 1.0,
        truncate: float = 4.0,
    ) -> np.ndarray:
        """Gaussian smoothing (isotropic or anisotropic).

        Uses :func:`scipy.ndimage.gaussian_filter`.
        """
        from scipy.ndimage import gaussian_filter

        if np.isscalar(sigma):
            sig = (sigma, sigma, sigma)
        else:
            sig = tuple(sigma)
            if len(sig) != 3:
                raise ValueError("sigma must be a scalar or a 3-tuple.")
        return gaussian_filter(
            np.asarray(volume, dtype=np.float32),
            sigma=sig,
            truncate=truncate,
            mode="reflect",
        ).astype(np.float32)

    @staticmethod
    def median_filter(volume: np.ndarray, size: int = 3) -> np.ndarray:
        """3D median filter (edge-preserving, removes salt-and-pepper noise)."""
        from scipy.ndimage import median_filter

        return median_filter(
            np.asarray(volume, dtype=np.float32),
            size=size,
            mode="reflect",
        ).astype(np.float32)

    @staticmethod
    def denoise_bilateral(
        volume: np.ndarray,
        sigma_spatial: float = 1.0,
        intensity_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Slice-wise bilateral denoising.

        ``skimage.restoration.denoise_bilateral`` is 2D-only, so we apply it
        slice-by-slice along the Z axis.  This is a common compromise for 3D
        medical volumes.
        """
        from skimage.restoration import denoise_bilateral

        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return vol
        if intensity_range is None:
            lo, hi = float(vol.min()), float(vol.max())
        else:
            lo, hi = float(intensity_range[0]), float(intensity_range[1])
        rng = max(hi - lo, 1e-6)
        out = np.empty_like(vol)
        for z in range(vol.shape[2]):
            out[:, :, z] = denoise_bilateral(
                vol[:, :, z],
                sigma_color=rng * 0.1,
                sigma_spatial=sigma_spatial,
                channel_axis=None,
            )
        return out.astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Bias field correction (N4)                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def bias_field_correction(
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        shrink_factor: int = 2,
        n_iterations: Tuple[int, int, int, int] = (50, 50, 30, 20),
        convergence_threshold: float = 1e-6,
    ) -> np.ndarray:
        """N4 bias-field correction via SimpleITK.

        Corrects the low-frequency intensity non-uniformity typical of MRI
        acquisitions.  Operationally robust on float volumes with non-zero
        positive values; the input is shifted to be strictly positive.

        Parameters
        ----------
        volume : np.ndarray
            3D input volume.
        mask : np.ndarray, optional
            Binary mask restricting the correction region.  If ``None``, the
            whole volume is used.
        shrink_factor : int
            Shrink factor for the multi-resolution pyramid (>=1).
        n_iterations : tuple of int
            Iterations per resolution level (length 4 recommended).
        convergence_threshold : float
            Convergence threshold on the log-magnitude of the bias field.

        Returns
        -------
        np.ndarray
            Bias-corrected volume, same shape as input.
        """
        import SimpleITK as sitk

        vol = np.asarray(volume, dtype=np.float32)
        if vol.size == 0:
            return vol

        # N4 requires strictly positive values.
        offset = 1.0
        if vol.min() <= 0:
            offset = float(-vol.min()) + 1.0
        vol_pos = vol + offset

        sitk_vol = sitk.GetImageFromArray(vol_pos)
        if mask is not None:
            mask_arr = (np.asarray(mask) > 0).astype(np.uint8)
            sitk_mask = sitk.GetImageFromArray(mask_arr)
            sitk_mask.CopyInformation(sitk_vol)
        else:
            otsu = sitk.OtsuThresholdImageFilter()
            sitk_mask = otsu.Execute(sitk_vol)

        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([int(i) for i in n_iterations])
        corrector.SetConvergenceThreshold(float(convergence_threshold))
        # SetShrinkFactor is available on the filter itself (newer SimpleITK)
        # and supersedes the deprecated 4th positional argument to Execute().
        if hasattr(corrector, "SetShrinkFactor"):
            try:
                corrector.SetShrinkFactor(int(shrink_factor))
            except (TypeError, AttributeError):
                pass

        try:
            # New API: Execute(image, mask) only. Use setters for the rest.
            corrector.Execute(sitk_vol, sitk_mask)
            log_bias = corrector.GetLogBiasFieldAsImage(sitk_vol)
            corrected = sitk_vol / sitk.Exp(log_bias)
            # GetArrayFromImage returns the array in the same axis order it
            # was given to GetImageFromArray — no transpose needed.
            out = sitk.GetArrayFromImage(corrected).astype(np.float32)
        except (RuntimeError, TypeError):
            # Fallback: try the older 4-arg API, then the mask-less API,
            # and finally return the shifted volume if everything fails.
            try:
                try:
                    corrector.Execute(sitk_vol, sitk_mask, True, int(shrink_factor))
                except TypeError:
                    corrector.Execute(sitk_vol)
                log_bias = corrector.GetLogBiasFieldAsImage(sitk_vol)
                corrected = sitk_vol / sitk.Exp(log_bias)
                out = sitk.GetArrayFromImage(corrected).astype(np.float32)
            except RuntimeError:
                out = vol_pos

        # Safety check: ensure shape is preserved.
        if out.shape != vol.shape:
            # Try to fix axis order if SimpleITK reshuffled them.
            try:
                out = out.transpose(np.argsort(np.argsort(vol.shape)))
            except Exception:
                out = vol_pos

        return out - offset
