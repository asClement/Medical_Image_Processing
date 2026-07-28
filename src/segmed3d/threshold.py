"""Threshold-based segmentation (Otsu, multi-Otsu, slice-wise Otsu)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseSegmentation

__all__ = ["ThresholdSegmentation"]


class ThresholdSegmentation(BaseSegmentation):
    """Threshold-based 3D segmentation.

    Supports three strategies:

    - ``'otsu'``: global Otsu threshold over the full 3D volume.
    - ``'multi_otsu'``: multi-class Otsu, the foreground class is selected by
      intensity (the brightest class by default).
    - ``'slice_wise'``: per-slice Otsu along the Z axis (useful for volumes
      with strong intensity drift).

    Examples
    --------
    >>> seg = ThresholdSegmentation(volume, affine, header)
    >>> seg.fit(method='otsu')
    >>> mask = seg.get_mask()
    >>> seg.save('mask.nii.gz')
    """

    def fit(
        self,
        method: str = "otsu",
        n_classes: int = 2,
        slice_wise: bool = False,
        nbins: int = 256,
    ) -> "ThresholdSegmentation":
        """Compute the threshold mask.

        Parameters
        ----------
        method : str, {'otsu', 'multi_otsu'}
            Thresholding strategy.
        n_classes : int
            Number of classes for ``'multi_otsu'`` (>=2).  For ``'otsu'`` the
            value is ignored (always 2 classes).
        slice_wise : bool
            If ``True``, the Otsu threshold is computed independently on each
            Z-slice.  Useful when intensity inhomogeneity is severe.
        nbins : int
            Number of histogram bins used by Otsu.

        Returns
        -------
        self
        """
        from skimage.filters import threshold_otsu
        # Handle the skimage 0.19 / 0.24+ rename: threshold_multi_otsu -> threshold_multiotsu
        try:
            from skimage.filters import threshold_multi_otsu as _multi_otsu
        except ImportError:
            from skimage.filters import threshold_multiotsu as _multi_otsu

        if slice_wise and method == "multi_otsu":
            raise ValueError("slice_wise is only supported with method='otsu'.")

        vol = self.volume_
        mask = np.zeros_like(vol, dtype=np.uint8)

        if method == "otsu":
            if slice_wise:
                for z in range(vol.shape[2]):
                    slc = vol[:, :, z]
                    if slc.max() - slc.min() < 1e-6:
                        continue
                    try:
                        t = threshold_otsu(slc, nbins=nbins)
                        mask[:, :, z] = (slc > t).astype(np.uint8)
                    except ValueError:
                        continue
                self._threshold_value_ = None
            else:
                t = threshold_otsu(vol, nbins=nbins)
                mask = (vol > t).astype(np.uint8)
                self._threshold_value_ = float(t)

        elif method == "multi_otsu":
            if n_classes < 2:
                raise ValueError("n_classes must be >= 2 for multi_otsu.")
            thresholds = _multi_otsu(vol, classes=n_classes, nbins=nbins)
            # Foreground = voxels above the highest threshold.
            mask = (vol > thresholds[-1]).astype(np.uint8)
            self._threshold_value_ = [float(t) for t in thresholds]

        else:
            raise ValueError(
                f"Unknown method {method!r}. Use 'otsu' or 'multi_otsu'."
            )

        self.mask_ = mask
        self._fitted = True
        self._method = method
        return self

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    def get_threshold(self):
        """Return the computed threshold(s).

        Returns
        -------
        float or list of float or None
            - ``'otsu'`` global: a single float.
            - ``'otsu'`` slice_wise: ``None`` (per-slice).
            - ``'multi_otsu'``: list of thresholds.
        """
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return getattr(self, "_threshold_value_", None)
