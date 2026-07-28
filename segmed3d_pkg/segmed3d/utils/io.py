"""NIfTI image I/O utilities.

This module wraps :mod:`nibabel` to provide a small, consistent API for
loading and saving 3D medical images and binary masks in the NIfTI format
(``.nii`` / ``.nii.gz``).
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

__all__ = ["ImageIO"]


class ImageIO:
    """Static helper class for NIfTI file I/O.

    All methods are ``@staticmethod`` — :class:`ImageIO` is never
    instantiated.  It serves purely as a namespace.

    Examples
    --------
    >>> vol, affine, header = ImageIO.load_nifti('tumor.nii.gz')
    >>> mask = (vol > vol.mean()).astype(np.uint8)
    >>> ImageIO.save_mask(mask, 'mask.nii.gz', affine=affine, header=header)
    """

    # ------------------------------------------------------------------ #
    #  Loading                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def load_nifti(path: str, dtype: Optional[type] = None) -> Tuple[np.ndarray, np.ndarray, object]:
        """Load a 3D NIfTI image.

        Parameters
        ----------
        path : str
            Path to a ``.nii`` or ``.nii.gz`` file.
        dtype : type, optional
            If given, cast the array to this dtype (e.g. ``np.float32``).

        Returns
        -------
        volume : np.ndarray
            3D array of shape ``(X, Y, Z)``.
        affine : np.ndarray
            4x4 affine matrix.
        header : nibabel.Nifti1Header
            File header.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the loaded array is not 3D.
        """
        import nibabel as nib

        if not os.path.isfile(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")

        img = nib.load(path)
        vol = np.asarray(img.get_fdata(), dtype=np.float32) if dtype is None \
            else np.asarray(img.get_fdata(), dtype=dtype)
        if vol.ndim != 3:
            raise ValueError(
                f"Expected a 3D NIfTI volume, got {vol.ndim}D with shape {vol.shape}."
            )
        affine = np.asarray(img.affine, dtype=np.float32)
        header = img.header
        return vol, affine, header

    @staticmethod
    def load_mask(path: str) -> Tuple[np.ndarray, np.ndarray, object]:
        """Load a binary segmentation mask from NIfTI.

        The returned mask is binarised (``> 0`` → 1) and cast to ``uint8``.

        Returns
        -------
        mask : np.ndarray
            Binary mask of shape ``(X, Y, Z)``, dtype ``uint8``.
        affine : np.ndarray
        header : object
        """
        vol, affine, header = ImageIO.load_nifti(path, dtype=np.float32)
        mask = (vol > 0).astype(np.uint8)
        return mask, affine, header

    # ------------------------------------------------------------------ #
    #  Saving                                                             #
    # ------------------------------------------------------------------ #
    @staticmethod
    def save_nifti(
        volume: np.ndarray,
        path: str,
        affine: Optional[np.ndarray] = None,
        header: Optional[object] = None,
    ) -> str:
        """Save a 3D array as a NIfTI file.

        Parameters
        ----------
        volume : np.ndarray
            3D array to save.
        path : str
            Output path.  Use ``.nii.gz`` for gzip compression.
        affine : np.ndarray, optional
            4x4 affine matrix.  Defaults to identity.
        header : nibabel header, optional
            Header to attach.  If ``None`` a default header is created.

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        import nibabel as nib

        if volume.ndim != 3:
            raise ValueError(f"volume must be 3D, got {volume.ndim}D.")
        affine = np.eye(4, dtype=np.float32) if affine is None \
            else np.asarray(affine, dtype=np.float32)
        img = nib.Nifti1Image(np.ascontiguousarray(volume), affine, header=header)
        abs_path = os.path.abspath(path)
        nib.save(img, abs_path)
        return abs_path

    @staticmethod
    def save_mask(
        mask: np.ndarray,
        path: str,
        affine: Optional[np.ndarray] = None,
        header: Optional[object] = None,
    ) -> str:
        """Save a binary mask as a NIfTI file (binarised, uint8)."""
        mask_bin = (np.asarray(mask) > 0).astype(np.uint8)
        return ImageIO.save_nifti(mask_bin, path, affine=affine, header=header)

    # ------------------------------------------------------------------ #
    #  Metadata helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def get_voxel_spacing(affine: np.ndarray) -> Tuple[float, float, float]:
        """Extract voxel spacing ``(sx, sy, sz)`` from an affine matrix."""
        a = np.asarray(affine, dtype=np.float64)
        sx = np.linalg.norm(a[:3, 0])
        sy = np.linalg.norm(a[:3, 1])
        sz = np.linalg.norm(a[:3, 2])
        return float(sx), float(sy), float(sz)
