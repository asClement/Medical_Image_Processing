"""Base abstract class for all segmentation methods in segmed3d.

This module defines :class:`BaseSegmentation`, the abstract base class (ABC)
that every segmentation algorithm in :mod:`segmed3d` inherits from.  The class
implements the common boilerplate (input validation, mask storage, NIfTI I/O,
``__call__`` shortcut) so that concrete subclasses only have to implement
:meth:`fit`.

The public API follows the *scikit-learn* style with a :meth:`fit` step that
computes the segmentation mask, followed by :meth:`get_mask` to retrieve the
result and :meth:`save` to persist it as a NIfTI file.

Example
-------
>>> from segmed3d import ThresholdSegmentation
>>> seg = ThresholdSegmentation(volume, affine, header)
>>> seg.fit(method='otsu')
>>> mask = seg.get_mask()
>>> seg.save('mask.nii.gz')
>>> # Or the one-liner equivalent:
>>> mask = ThresholdSegmentation(volume, affine, header)(method='otsu')
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

__all__ = ["BaseSegmentation"]


class BaseSegmentation(ABC):
    """Abstract base class for all 3D segmentation algorithms.

    Parameters
    ----------
    volume : np.ndarray
        3D image volume of shape ``(X, Y, Z)``.  Any real-valued numeric
        dtype is accepted; the array is internally cast to ``float32`` for
        numerical stability.
    affine : np.ndarray, optional
        4x4 affine transformation matrix mapping voxel coordinates to world
        coordinates.  If ``None``, an identity matrix is used.  This is
        required for correct NIfTI export.
    header : nibabel header, optional
        NIfTI header associated with ``volume``.  If ``None``, a default
        header is generated on save.

    Attributes
    ----------
    volume_ : np.ndarray
        Validated and float-cast copy of the input volume.
    affine_ : np.ndarray
        4x4 affine matrix.
    header_ : object
        NIfTI header (or ``None``).
    mask_ : np.ndarray or None
        Computed boolean/uint8 segmentation mask after :meth:`fit` is called.
        ``None`` before fitting.
    shape : tuple
        Shape of the input volume.
    """

    # ------------------------------------------------------------------ #
    #  Construction & validation                                          #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        volume: np.ndarray,
        affine: Optional[np.ndarray] = None,
        header: Optional[object] = None,
    ) -> None:
        self.volume_ = self._validate_volume(volume)
        self.shape = self.volume_.shape

        if affine is None:
            self.affine_ = np.eye(4, dtype=np.float32)
        else:
            affine = np.asarray(affine, dtype=np.float32)
            if affine.shape != (4, 4):
                raise ValueError(
                    f"affine must be a 4x4 matrix, got shape {affine.shape}."
                )
            self.affine_ = affine

        self.header_ = header
        self.mask_: Optional[np.ndarray] = None
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Validation                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_volume(volume: np.ndarray) -> np.ndarray:
        """Validate the input volume and return a float32 copy.

        Parameters
        ----------
        volume : np.ndarray
            Candidate input volume.

        Returns
        -------
        np.ndarray
            Validated volume as ``float32``.

        Raises
        ------
        TypeError
            If ``volume`` is not a NumPy array.
        ValueError
            If the array is not 3D, is empty, or contains non-finite values.
        """
        if not isinstance(volume, np.ndarray):
            raise TypeError(
                f"volume must be a numpy.ndarray, got {type(volume).__name__}."
            )
        if volume.ndim != 3:
            raise ValueError(
                f"volume must be 3D (X, Y, Z), got {volume.ndim}D with shape "
                f"{volume.shape}."
            )
        if volume.size == 0:
            raise ValueError("volume is empty (size == 0).")

        vol = volume.astype(np.float32, copy=True)
        if not np.all(np.isfinite(vol)):
            # Replace NaN/Inf with 0 to keep downstream algorithms safe.
            vol[~np.isfinite(vol)] = 0.0
        return vol

    # ------------------------------------------------------------------ #
    #  Abstract API                                                       #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fit(self, *args, **kwargs):
        """Compute the segmentation mask.

        Concrete subclasses must populate ``self.mask_`` with a 3D array of
        dtype ``uint8`` (values in ``{0, 1}``) or boolean, and set
        ``self._fitted = True`` before returning ``self``.
        """

    # ------------------------------------------------------------------ #
    #  Public helpers                                                     #
    # ------------------------------------------------------------------ #
    def get_mask(self) -> np.ndarray:
        """Return the computed segmentation mask.

        Returns
        -------
        np.ndarray
            Binary mask of shape ``self.shape``, dtype ``uint8``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._fitted or self.mask_ is None:
            raise RuntimeError(
                "The segmentation has not been fitted yet. "
                "Call `.fit(...)` before `.get_mask()`."
            )
        return self.mask_.astype(np.uint8, copy=False)

    def save(self, path: str) -> str:
        """Save the computed mask as a NIfTI file.

        Parameters
        ----------
        path : str
            Output path.  ``.nii`` or ``.nii.gz`` extensions are recommended.

        Returns
        -------
        str
            The absolute path to the saved file.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self._fitted or self.mask_ is None:
            raise RuntimeError(
                "Cannot save: call `.fit(...)` first to compute the mask."
            )
        # Local import to avoid a hard nibabel dependency at import time.
        from .utils.io import ImageIO

        return ImageIO.save_mask(self.get_mask(), path, affine=self.affine_,
                                 header=self.header_)

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Run :meth:`fit` and return the mask in one call.

        Equivalent to::

            seg.fit(*args, **kwargs)
            mask = seg.get_mask()
        """
        self.fit(*args, **kwargs)
        return self.get_mask()

    # ------------------------------------------------------------------ #
    #  Dunders                                                            #
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        fitted = "fitted" if self._fitted else "not fitted"
        return f"<{type(self).__name__} shape={self.shape} ({fitted})>"

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------ #
    #  Convenience copy/clone                                             #
    # ------------------------------------------------------------------ #
    def clone(self) -> "BaseSegmentation":
        """Return a deep copy of the segmenter (without the fitted mask)."""
        new = copy.copy(self)
        new.volume_ = self.volume_.copy()
        new.affine_ = self.affine_.copy()
        new.mask_ = None
        new._fitted = False
        return new
