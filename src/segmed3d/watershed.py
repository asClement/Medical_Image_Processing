"""Watershed segmentation for 3D volumes.

Implements gradient-based, distance-transform-based and marker-driven
watershed, all backed by :func:`skimage.segmentation.watershed`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseSegmentation

__all__ = ["WatershedSegmentation"]


class WatershedSegmentation(BaseSegmentation):
    """3D watershed segmentation.

    The watershed transform treats the volume as a topographic relief and
    floods basins from user-provided *markers*.  Three common flows are
    supported:

    - ``'gradient'``: classic watershed on the gradient magnitude.
    - ``'distance'``: watershed on the (negative) distance transform of a
      binary mask — ideal for separating touching objects.
    - ``'image'``: watershed directly on the intensity (or its inverse).

    Examples
    --------
    >>> from scipy.ndimage import label
    >>> markers = np.zeros_like(volume, dtype=np.int32)
    >>> markers[volume < volume.mean()] = 1   # background
    >>> markers[volume > volume.mean() + 2*volume.std()] = 2  # foreground
    >>> seg = WatershedSegmentation(volume, affine, header)
    >>> seg.fit(markers=markers, gradient_method='gradient')
    """

    def fit(
        self,
        markers: np.ndarray,
        gradient_method: str = "gradient",
        connectivity: int = 26,
        compactness: float = 0.0,
        mask: Optional[np.ndarray] = None,
        sigma: float = 1.0,
    ) -> "WatershedSegmentation":
        """Run the watershed transform.

        Parameters
        ----------
        markers : np.ndarray
            Integer label array (same shape as volume) seeding each basin.
            ``0`` means "no marker".  At least two distinct non-zero labels
            are required.
        gradient_method : str, {'gradient', 'distance', 'image'}
            - ``'gradient'``: Sobel gradient magnitude.
            - ``'distance'``: use the distance transform of ``markers>0`` as
              the elevation map.
            - ``'image'``: use the (inverted) volume directly.
        connectivity : int, {6, 18, 26}
            Voxel connectivity.
        compactness : float
            Compactness parameter (higher values produce more regular basins).
        mask : np.ndarray, optional
            Optional binary mask restricting the region where watershed is
            computed.
        sigma : float
            Gaussian smoothing applied to the volume before gradient
            computation (``gradient_method='gradient'`` only).

        Returns
        -------
        self
        """
        from scipy import ndimage as ndi
        from skimage.filters import sobel
        from skimage.segmentation import watershed

        markers = np.asarray(markers).astype(np.int32)
        if markers.shape != self.volume_.shape:
            raise ValueError(
                f"markers shape {markers.shape} != volume {self.volume_.shape}."
            )
        if markers.max() < 2:
            raise ValueError(
                "markers must contain at least 2 distinct non-zero labels."
            )

        if connectivity == 6:
            conn = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ])
        elif connectivity == 18:
            conn = np.array([
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            ])
        elif connectivity == 26:
            conn = np.ones((3, 3, 3), dtype=np.uint8)
        else:
            raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}.")

        if gradient_method == "gradient":
            from scipy.ndimage import gaussian_filter
            smoothed = gaussian_filter(self.volume_, sigma=sigma)
            elevation = sobel(smoothed)
        elif gradient_method == "distance":
            binary = (markers > 0).astype(np.uint8)
            # Distance transform inside each marker region.
            dt = ndi.distance_transform_edt(binary == 0)  # outside markers
            elevation = -dt  # markers are local minima
        elif gradient_method == "image":
            elevation = self.volume_.max() - self.volume_  # invert
        else:
            raise ValueError(
                f"Unknown gradient_method {gradient_method!r}. "
                "Use 'gradient', 'distance' or 'image'."
            )

        labels = watershed(
            elevation,
            markers=markers,
            connectivity=conn,
            compactness=compactness,
            mask=(np.asarray(mask) > 0) if mask is not None else None,
        )

        # Convert labels to binary mask: keep any non-background label
        # (label > 1 by convention; 1 = background).  If only label 1 exists
        # as "background", we keep all non-zero.
        if (markers == 1).any():
            mask_out = (labels > 1).astype(np.uint8)
        else:
            mask_out = (labels > 0).astype(np.uint8)

        self.mask_ = mask_out
        self._labels_ = labels
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Accessor                                                           #
    # ------------------------------------------------------------------ #
    def get_labels(self) -> np.ndarray:
        """Return the full label map (one int per basin)."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._labels_.copy()
