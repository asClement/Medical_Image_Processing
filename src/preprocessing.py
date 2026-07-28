
from __future__ import annotations

import importlib
from typing import Literal, Sequence

import nibabel as nib
import numpy as np
import SimpleITK as sitk


class Preprocessing:
    """Collection of intensity preprocessing methods for 3D NIfTI volumes.

    Design principles:
    - Every public method takes a NIfTI path as first argument.
    - Every public method returns a ``nib.Nifti1Image`` object.
    - Statistics are computed on foreground only by default (voxels > 0.0).
    - Background is preserved unchanged unless explicitly requested.
    """

    def __init__(self) -> None:
        """Create a preprocessing utility instance."""
        pass

    @staticmethod
    def _load_nifti(nifti_path: str) -> tuple[nib.Nifti1Image, np.ndarray]:
        """Load a NIfTI image and return image object + float32 data array."""
        img = nib.load(nifti_path)
        data = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
        return img, data

    @staticmethod
    def _foreground_mask(data: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Return a boolean foreground mask using ``data > threshold``."""
        return data > threshold

    @staticmethod
    def _as_nifti(
        data: np.ndarray,
        reference_img: nib.Nifti1Image,
        dtype: np.dtype = np.float32,
    ) -> nib.Nifti1Image:
        """Build a NIfTI image from ``data`` while preserving affine/header."""
        out = np.asarray(data, dtype=dtype)
        return nib.Nifti1Image(out, affine=reference_img.affine, header=reference_img.header)

    @staticmethod
    def _validate_non_empty_mask(mask: np.ndarray, method_name: str) -> None:
        """Raise a clear error if no foreground voxel is available."""
        if not np.any(mask):
            raise ValueError(
                f"{method_name}: foreground mask is empty. "
                "Adjust foreground_threshold or check image content."
            )

    def min_max_global_scaling(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        output_range: tuple[float, float] = (0.0, 1.0),
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply global min-max scaling on foreground intensities.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels with intensity > threshold are used.
        output_range : tuple[float, float], default=(0.0, 1.0)
            Target interval ``(a, b)`` for scaled values.
        preserve_background : bool, default=True
            If True, voxels outside foreground keep original values.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with scaled foreground.

        Notes
        -----
        Scaling formula on foreground voxels:
        ``x_scaled = a + (x - min_fg) * (b - a) / (max_fg - min_fg)``.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.min_max_global_scaling("subject.nii.gz")
        >>> out_data = out_img.get_fdata()
        """
        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "min_max_global_scaling")

        a, b = output_range
        if not b > a:
            raise ValueError("output_range must satisfy max > min.")

        fg = data[mask]
        fg_min = float(np.min(fg))
        fg_max = float(np.max(fg))
        denom = fg_max - fg_min

        out = data.copy()
        if denom <= 0.0:
            out[mask] = a
        else:
            out[mask] = a + (fg - fg_min) * (b - a) / denom

        if not preserve_background:
            out[~mask] = a

        return self._as_nifti(out, img)

    def robust_min_max_scaling(
        self,
        nifti_path: str,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        foreground_threshold: float = 0.0,
        output_range: tuple[float, float] = (0.0, 1.0),
        clip: bool = True,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply robust min-max scaling using percentile anchors.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        lower_percentile : float, default=1.0
            Lower percentile used as robust minimum.
        upper_percentile : float, default=99.0
            Upper percentile used as robust maximum.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        output_range : tuple[float, float], default=(0.0, 1.0)
            Target output interval.
        clip : bool, default=True
            If True, values outside percentile range are clipped before scaling.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with robustly scaled foreground.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.robust_min_max_scaling("subject.nii.gz", 2, 98)
        """
        if not (0.0 <= lower_percentile < upper_percentile <= 100.0):
            raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "robust_min_max_scaling")

        a, b = output_range
        if not b > a:
            raise ValueError("output_range must satisfy max > min.")

        fg = data[mask]
        p_low, p_high = np.percentile(fg, [lower_percentile, upper_percentile])
        denom = float(p_high - p_low)

        out = data.copy()
        work = fg.copy()
        if clip:
            work = np.clip(work, p_low, p_high)

        if denom <= 0.0:
            out[mask] = a
        else:
            out[mask] = a + (work - p_low) * (b - a) / denom

        if not preserve_background:
            out[~mask] = a

        return self._as_nifti(out, img)

    def z_score_global_normalization(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        eps: float = 1e-8,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply global z-score normalization on foreground intensities.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image where foreground is standardized with mean/std.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.z_score_global_normalization("subject.nii.gz")
        """
        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "z_score_global_normalization")

        fg = data[mask]
        mu = float(np.mean(fg))
        sigma = float(np.std(fg))

        out = data.copy()
        out[mask] = (fg - mu) / max(sigma, eps)

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def robust_z_score_normalization(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        eps: float = 1e-8,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply robust z-score normalization using median and IQR.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            Foreground standardized with robust location/scale.

        Notes
        -----
        Uses:
        - location = median(foreground)
        - robust_std = IQR / 1.349, with IQR = Q3 - Q1

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.robust_z_score_normalization("subject.nii.gz")
        """
        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "robust_z_score_normalization")

        fg = data[mask]
        med = float(np.median(fg))
        q1, q3 = np.percentile(fg, [25.0, 75.0])
        robust_std = float((q3 - q1) / 1.349)

        out = data.copy()
        out[mask] = (fg - med) / max(robust_std, eps)

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def median_mad_scaling(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        eps: float = 1e-8,
        normal_consistency: bool = True,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Scale intensities with median and MAD (Median Absolute Deviation).

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        eps : float, default=1e-8
            Small constant to avoid division by zero.
        normal_consistency : bool, default=True
            If True, MAD is multiplied by 1.4826 for normal-consistent scale.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with robust median/MAD scaling.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.median_mad_scaling("subject.nii.gz")
        """
        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "median_mad_scaling")

        fg = data[mask]
        med = float(np.median(fg))
        mad = float(np.median(np.abs(fg - med)))
        if normal_consistency:
            mad *= 1.4826

        out = data.copy()
        out[mask] = (fg - med) / max(mad, eps)

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def adaptive_histogram_equalization(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        alpha: float = 0.3,
        beta: float = 0.3,
        radius: Sequence[int] = (8, 8, 4),
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply 3D Adaptive Histogram Equalization (AHE) using SimpleITK.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        alpha : float, default=0.3
            AHE alpha parameter in SimpleITK.
        beta : float, default=0.3
            AHE beta parameter in SimpleITK.
        radius : Sequence[int], default=(8, 8, 4)
            Neighborhood radius for local histogram operations.
        preserve_background : bool, default=True
            If True, background is restored from original image.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image after local contrast enhancement.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.adaptive_histogram_equalization("subject.nii.gz", radius=(6, 6, 3))
        """
        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "adaptive_histogram_equalization")

        sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
        zooms = img.header.get_zooms()[:3]
        if len(zooms) == 3:
            sitk_img.SetSpacing(tuple(float(z) for z in zooms))

        ahe = sitk.AdaptiveHistogramEqualizationImageFilter()
        ahe.SetAlpha(float(alpha))
        ahe.SetBeta(float(beta))
        ahe.SetRadius([int(r) for r in radius])
        enhanced = ahe.Execute(sitk_img)

        out = sitk.GetArrayFromImage(enhanced).astype(np.float32)
        if preserve_background:
            out[~mask] = data[~mask]

        return self._as_nifti(out, img)

    def clahe(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        clip_limit: float = 0.01,
        nbins: int = 256,
        kernel_size: tuple[int, int] | None = None,
        axis: int = 2,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply CLAHE slice-wise using scikit-image.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        clip_limit : float, default=0.01
            CLAHE clipping limit in ``skimage.exposure.equalize_adapthist``.
        nbins : int, default=256
            Number of bins for local histograms.
        kernel_size : tuple[int, int] | None, default=None
            Tile size for each 2D slice. If None, skimage default is used.
        axis : int, default=2
            Slice axis for 3D processing.
        preserve_background : bool, default=True
            If True, background is restored from original image.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with CLAHE-enhanced foreground.

        Raises
        ------
        ImportError
            If scikit-image is not installed.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.clahe("subject.nii.gz", clip_limit=0.02, kernel_size=(32, 32))
        """
        try:
            exposure = importlib.import_module("skimage.exposure")
        except ImportError as exc:
            raise ImportError(
                "CLAHE requires scikit-image. Install it with: pip install scikit-image"
            ) from exc

        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2 for 3D volumes.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "clahe")

        fg = data[mask]
        fg_min = float(np.min(fg))
        fg_max = float(np.max(fg))
        denom = max(fg_max - fg_min, 1e-8)

        norm = data.copy()
        norm[mask] = (fg - fg_min) / denom

        moved = np.moveaxis(norm, axis, 0)
        moved_mask = np.moveaxis(mask, axis, 0)
        out_moved = moved.copy()

        for i in range(moved.shape[0]):
            sl = moved[i]
            sl_mask = moved_mask[i]
            if not np.any(sl_mask):
                continue
            # CLAHE is applied on full slice in [0, 1], then masked.
            sl_eq = exposure.equalize_adapthist(sl, kernel_size=kernel_size, clip_limit=clip_limit, nbins=nbins)
            out_moved[i, sl_mask] = sl_eq[sl_mask]

        out_norm = np.moveaxis(out_moved, 0, axis)
        out = data.copy()
        out[mask] = out_norm[mask] * denom + fg_min

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def histogram_equalization(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        nbins: int = 1024,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply global histogram equalization on foreground only.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        nbins : int, default=1024
            Number of bins to build the empirical CDF.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with CDF-mapped foreground values.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.histogram_equalization("subject.nii.gz", nbins=512)
        """
        if nbins < 2:
            raise ValueError("nbins must be >= 2.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "histogram_equalization")

        fg = data[mask]
        hist, bin_edges = np.histogram(fg, bins=int(nbins), density=False)
        cdf = hist.cumsum().astype(np.float64)
        cdf /= cdf[-1]

        # Map voxel values through CDF interpolation.
        x = fg.astype(np.float64)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        mapped = np.interp(x, bin_centers, cdf)

        out = data.copy()
        out[mask] = mapped.astype(np.float32)

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def gamma_correction(
        self,
        nifti_path: str,
        gamma: float = 1.0,
        foreground_threshold: float = 0.0,
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply gamma correction on foreground with data-driven normalization.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        gamma : float, default=1.0
            Gamma exponent. ``gamma < 1`` brightens, ``gamma > 1`` darkens.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image with gamma-adjusted foreground.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.gamma_correction("subject.nii.gz", gamma=0.8)
        """
        if gamma <= 0.0:
            raise ValueError("gamma must be > 0.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "gamma_correction")

        fg = data[mask]
        fg_min = float(np.min(fg))
        fg_max = float(np.max(fg))
        denom = max(fg_max - fg_min, 1e-8)

        norm = (fg - fg_min) / denom
        corrected = np.power(norm, gamma)

        out = data.copy()
        out[mask] = corrected * denom + fg_min

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out, img)

    def bias_field_correction(
        self,
        nifti_path: str,
        foreground_threshold: float = 0.0,
        shrink_factor: int = 2,
        spline_order: int = 3,
        n_fitting_levels: int = 4,
        n_iterations_per_level: tuple[int, ...] = (50, 50, 30, 20),
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Correct intensity inhomogeneity with N4 bias field correction.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        shrink_factor : int, default=2
            Downsampling factor for faster N4 estimation.
        spline_order : int, default=3
            B-spline order used by N4.
        n_fitting_levels : int, default=4
            Number of fitting levels.
        n_iterations_per_level : tuple[int, ...], default=(50, 50, 30, 20)
            Iterations per fitting level.
        preserve_background : bool, default=True
            If True, background is restored from original image.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image corrected for low-frequency bias field.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.bias_field_correction("subject.nii.gz", shrink_factor=4)
        """
        if shrink_factor < 1:
            raise ValueError("shrink_factor must be >= 1.")
        if n_fitting_levels < 1:
            raise ValueError("n_fitting_levels must be >= 1.")
        if len(n_iterations_per_level) != n_fitting_levels:
            raise ValueError("len(n_iterations_per_level) must match n_fitting_levels.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "bias_field_correction")

        sitk_img = sitk.GetImageFromArray(data.astype(np.float32))
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))

        zooms = img.header.get_zooms()[:3]
        if len(zooms) == 3:
            spacing = tuple(float(z) for z in zooms)
            sitk_img.SetSpacing(spacing)
            sitk_mask.SetSpacing(spacing)

        if shrink_factor > 1:
            img_small = sitk.Shrink(sitk_img, [shrink_factor] * sitk_img.GetDimension())
            mask_small = sitk.Shrink(sitk_mask, [shrink_factor] * sitk_mask.GetDimension())
        else:
            img_small = sitk_img
            mask_small = sitk_mask

        n4 = sitk.N4BiasFieldCorrectionImageFilter()
        n4.SetSplineOrder(int(spline_order))
        n4.SetMaximumNumberOfIterations([int(v) for v in n_iterations_per_level])

        _ = n4.Execute(img_small, mask_small)
        log_bias = n4.GetLogBiasFieldAsImage(sitk_img)
        corrected = sitk_img / sitk.Exp(log_bias)

        out = sitk.GetArrayFromImage(corrected).astype(np.float32)
        if preserve_background:
            out[~mask] = data[~mask]

        return self._as_nifti(out, img)

    def gaussian_mixture_model(
        self,
        nifti_path: str,
        n_components: int = 3,
        foreground_threshold: float = 0.0,
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = 42,
        return_mode: Literal["expected_mean", "hard_labels", "brightest_posterior"] = "expected_mean",
        preserve_background: bool = True,
    ) -> nib.Nifti1Image:
        """Apply a 1D Gaussian Mixture Model (EM) on foreground intensities.

        Parameters
        ----------
        nifti_path : str
            Path to the input NIfTI file.
        n_components : int, default=3
            Number of Gaussian components.
        foreground_threshold : float, default=0.0
            Foreground mask rule: voxels > threshold.
        max_iter : int, default=200
            Maximum EM iterations.
        tol : float, default=1e-4
            Convergence tolerance on average log-likelihood.
        random_state : int | None, default=42
            Random seed for initialization.
        return_mode : {"expected_mean", "hard_labels", "brightest_posterior"}, default="expected_mean"
            - ``expected_mean``: each voxel gets posterior-weighted component mean.
            - ``hard_labels``: each voxel gets discrete label 1..K.
            - ``brightest_posterior``: posterior probability of brightest-mean component.
        preserve_background : bool, default=True
            If True, background is left unchanged.

        Returns
        -------
        nib.Nifti1Image
            NIfTI image derived from fitted GMM on foreground.

        Notes
        -----
        This implementation is dependency-light and uses NumPy EM on 1D intensities.

        Example
        -------
        >>> p = Preprocessing()
        >>> out_img = p.gaussian_mixture_model("subject.nii.gz", n_components=4)
        """
        if n_components < 1:
            raise ValueError("n_components must be >= 1.")
        if max_iter < 1:
            raise ValueError("max_iter must be >= 1.")

        img, data = self._load_nifti(nifti_path)
        mask = self._foreground_mask(data, foreground_threshold)
        self._validate_non_empty_mask(mask, "gaussian_mixture_model")

        x = data[mask].astype(np.float64)
        n = x.size
        k = int(n_components)

        rng = np.random.default_rng(random_state)
        if n < k:
            raise ValueError("Number of foreground voxels is smaller than n_components.")

        # Initialization from quantiles for stable 1D mixture start.
        quantiles = np.linspace(0.0, 100.0, num=k + 2)[1:-1]
        means = np.percentile(x, quantiles).astype(np.float64)
        variances = np.full(k, np.var(x) + 1e-6, dtype=np.float64)
        weights = np.full(k, 1.0 / k, dtype=np.float64)

        prev_ll = -np.inf
        eps = 1e-12

        for _ in range(max_iter):
            # E-step
            diff = x[:, None] - means[None, :]
            inv_var = 1.0 / np.maximum(variances, eps)
            log_prob = -0.5 * (np.log(2.0 * np.pi) + np.log(np.maximum(variances, eps)) + (diff * diff) * inv_var)
            log_weighted = log_prob + np.log(np.maximum(weights, eps))[None, :]

            max_log = np.max(log_weighted, axis=1, keepdims=True)
            stabilized = np.exp(log_weighted - max_log)
            denom = np.sum(stabilized, axis=1, keepdims=True)
            resp = stabilized / np.maximum(denom, eps)

            # M-step
            nk = np.sum(resp, axis=0) + eps
            weights = nk / float(n)
            means = np.sum(resp * x[:, None], axis=0) / nk
            diff2 = x[:, None] - means[None, :]
            variances = np.sum(resp * (diff2 * diff2), axis=0) / nk
            variances = np.maximum(variances, 1e-6)

            ll = float(np.mean(max_log + np.log(np.maximum(denom, eps))))
            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        # Stable ordering by increasing component mean.
        order = np.argsort(means)
        means = means[order]
        weights = weights[order]
        variances = variances[order]

        # Recompute responsibilities with ordered parameters.
        diff = x[:, None] - means[None, :]
        inv_var = 1.0 / np.maximum(variances, eps)
        log_prob = -0.5 * (np.log(2.0 * np.pi) + np.log(np.maximum(variances, eps)) + (diff * diff) * inv_var)
        log_weighted = log_prob + np.log(np.maximum(weights, eps))[None, :]
        max_log = np.max(log_weighted, axis=1, keepdims=True)
        stabilized = np.exp(log_weighted - max_log)
        resp = stabilized / np.maximum(np.sum(stabilized, axis=1, keepdims=True), eps)
        hard = np.argmax(resp, axis=1)

        if return_mode == "expected_mean":
            fg_out = np.sum(resp * means[None, :], axis=1)
        elif return_mode == "hard_labels":
            fg_out = (hard + 1).astype(np.float64)
        elif return_mode == "brightest_posterior":
            fg_out = resp[:, -1]
        else:
            raise ValueError("Invalid return_mode.")

        out = data.copy().astype(np.float64)
        out[mask] = fg_out

        if not preserve_background:
            out[~mask] = 0.0

        return self._as_nifti(out.astype(np.float32), img)

    