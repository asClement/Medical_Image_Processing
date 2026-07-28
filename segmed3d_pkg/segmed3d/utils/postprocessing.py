"""Post-processing utilities for binary 3D segmentation masks.

Provides mask cleaning, hole filling, connected-component analysis and
classical morphological operations (erosion / dilation / opening / closing).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["Postprocessor"]


class Postprocessor:
    """Static collection of mask post-processing routines."""

    # ------------------------------------------------------------------ #
    #  Geometry cleanup                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def fill_holes(mask: np.ndarray, connectivity: int = 26) -> np.ndarray:
        """Fill interior holes in a binary 3D mask.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask.
        connectivity : int, {6, 18, 26}
            Connectivity used to identify background holes.

        Returns
        -------
        np.ndarray
            Hole-filled mask, dtype ``uint8``.
        """
        from scipy.ndimage import binary_fill_holes

        m = np.asarray(mask) > 0
        structure = Postprocessor._structure(connectivity)
        filled = binary_fill_holes(m, structure=structure)
        return filled.astype(np.uint8)

    @staticmethod
    def remove_small_objects(mask: np.ndarray, min_size: int = 50,
                             connectivity: int = 26) -> np.ndarray:
        """Remove connected components smaller than ``min_size`` voxels."""
        from scipy.ndimage import label

        m = np.asarray(mask) > 0
        if not m.any() or min_size <= 0:
            return m.astype(np.uint8)
        structure = Postprocessor._structure(connectivity)
        lbl, n = label(m, structure=structure)
        if n == 0:
            return np.zeros_like(m, dtype=np.uint8)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0  # background
        keep = counts >= min_size
        out = keep[lbl]
        return out.astype(np.uint8)

    @staticmethod
    def largest_cc(mask: np.ndarray, connectivity: int = 26) -> np.ndarray:
        """Keep only the largest connected component of the mask."""
        from scipy.ndimage import label

        m = np.asarray(mask) > 0
        if not m.any():
            return np.zeros_like(m, dtype=np.uint8)
        structure = Postprocessor._structure(connectivity)
        lbl, n = label(m, structure=structure)
        if n == 0:
            return np.zeros_like(m, dtype=np.uint8)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        target = int(np.argmax(counts))
        return (lbl == target).astype(np.uint8)

    @staticmethod
    def extract_largest_n(mask: np.ndarray, n: int = 2,
                          connectivity: int = 26) -> np.ndarray:
        """Keep the ``n`` largest connected components."""
        from scipy.ndimage import label

        m = np.asarray(mask) > 0
        if not m.any() or n <= 0:
            return np.zeros_like(m, dtype=np.uint8)
        structure = Postprocessor._structure(connectivity)
        lbl, num = label(m, structure=structure)
        if num == 0:
            return np.zeros_like(m, dtype=np.uint8)
        counts = np.bincount(lbl.ravel())
        counts[0] = 0
        order = np.argsort(counts)[::-1]
        keep_ids = order[1:n + 1]  # exclude background (0)
        out = np.isin(lbl, keep_ids)
        return out.astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Pipeline helper                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def clean_mask(
        mask: np.ndarray,
        min_size: int = 50,
        fill_holes: bool = True,
        connectivity: int = 26,
        keep_largest: bool = False,
    ) -> np.ndarray:
        """All-in-one mask cleanup pipeline.

        Steps
        -----
        1. Binarise.
        2. (optional) Fill holes.
        3. Remove small objects < ``min_size`` voxels.
        4. (optional) Keep only the largest connected component.

        Returns
        -------
        np.ndarray
            Cleaned binary mask, dtype ``uint8``.
        """
        m = np.asarray(mask) > 0
        if not m.any():
            return np.zeros_like(m, dtype=np.uint8)
        if fill_holes:
            m = (Postprocessor.fill_holes(m.astype(np.uint8), connectivity) > 0)
        m = Postprocessor.remove_small_objects(m.astype(np.uint8), min_size, connectivity) > 0
        if keep_largest:
            m = Postprocessor.largest_cc(m.astype(np.uint8), connectivity) > 0
        return m.astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Morphological operations                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def morph_open(mask: np.ndarray, radius: int = 1,
                   connectivity: int = 26) -> np.ndarray:
        """Binary opening (erosion followed by dilation)."""
        from scipy.ndimage import binary_opening

        m = np.asarray(mask) > 0
        s = Postprocessor._ball(radius) if radius > 0 else \
            Postprocessor._structure(connectivity)
        return binary_opening(m, structure=s).astype(np.uint8)

    @staticmethod
    def morph_close(mask: np.ndarray, radius: int = 1,
                    connectivity: int = 26) -> np.ndarray:
        """Binary closing (dilation followed by erosion)."""
        from scipy.ndimage import binary_closing

        m = np.asarray(mask) > 0
        s = Postprocessor._ball(radius) if radius > 0 else \
            Postprocessor._structure(connectivity)
        return binary_closing(m, structure=s).astype(np.uint8)

    @staticmethod
    def morph_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Binary erosion with a spherical structuring element."""
        from scipy.ndimage import binary_erosion

        m = np.asarray(mask) > 0
        s = Postprocessor._ball(radius)
        return binary_erosion(m, structure=s).astype(np.uint8)

    @staticmethod
    def morph_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
        """Binary dilation with a spherical structuring element."""
        from scipy.ndimage import binary_dilation

        m = np.asarray(mask) > 0
        s = Postprocessor._ball(radius)
        return binary_dilation(m, structure=s).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _structure(connectivity: int) -> np.ndarray:
        if connectivity == 6:
            return np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ])
        elif connectivity == 18:
            return np.array([
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            ])
        elif connectivity == 26:
            return np.ones((3, 3, 3), dtype=np.uint8)
        else:
            raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}.")

    @staticmethod
    def _ball(radius: int) -> np.ndarray:
        """Spherical structuring element of given integer radius."""
        if radius < 1:
            return np.ones((1, 1, 1), dtype=np.uint8)
        size = 2 * radius + 1
        z, y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1, -radius:radius + 1]
        d2 = x * x + y * y + z * z
        return (d2 <= radius * radius).astype(np.uint8)
