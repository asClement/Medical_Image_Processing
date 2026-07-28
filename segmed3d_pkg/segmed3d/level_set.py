"""Level-set segmentation for 3D volumes.

Provides two formulations:

- ``'chan_vese'``: classic Chan-Vese region-based level set, applied
  slice-wise (``skimage.segmentation.chan_vese`` is 2D-only) and re-stacked.
- ``'morphological_geodesic'``: thin-plate morphological level set (true 3D)
  via :func:`skimage.segmentation.morphological_chan_vese`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseSegmentation

__all__ = ["LevelSetSegmentation"]


class LevelSetSegmentation(BaseSegmentation):
    """3D level-set segmentation.

    Examples
    --------
    >>> init = np.zeros_like(volume, dtype=np.uint8)
    >>> init[40:60, 40:60, 40:60] = 1
    >>> seg = LevelSetSegmentation(volume)
    >>> seg.fit(init_mask=init, method='morphological_chan_vese', iterations=50)
    """

    def fit(
        self,
        init_mask: np.ndarray,
        method: str = "morphological_chan_vese",
        iterations: int = 50,
        dt: float = 0.5,
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        smoothing: int = 1,
        threshold: str = "auto",
        balloon: float = 0.0,
    ) -> "LevelSetSegmentation":
        """Run the level-set evolution.

        Parameters
        ----------
        init_mask : np.ndarray
            Initial binary mask (same shape as volume).
        method : str, {'chan_vese', 'morphological_chan_vese', 'morphological_geodesic'}
            - ``'chan_vese'``: 2D slice-wise Chan-Vese (region-based).
            - ``'morphological_chan_vese'``: 3D morphological ACWE (fast, true 3D).
            - ``'morphological_geodesic'``: 3D morphological GAC (edge-based).
        iterations : int
            Number of evolution iterations.
        dt : float
            Time step (chan_vese only).
        lambda1, lambda2 : float
            Weight of inside / outside region energy (chan_vese only).
        smoothing : int
            Number of smoothing repetitions per iteration (morphological methods).
        threshold : str or float
            Edge threshold (morphological_geodesic only).
        balloon : float
            Balloon force (morphological_geodesic only).

        Returns
        -------
        self
        """
        init = (np.asarray(init_mask) > 0).astype(np.float32)
        if init.shape != self.volume_.shape:
            raise ValueError(
                f"init_mask shape {init.shape} != volume {self.volume_.shape}."
            )

        if method == "chan_vese":
            mask = self._run_chan_vese_slice_wise(
                init, iterations, dt, lambda1, lambda2
            )
        elif method == "morphological_chan_vese":
            mask = self._run_morphological_cv(init, iterations, smoothing, lambda1, lambda2)
        elif method == "morphological_geodesic":
            mask = self._run_morphological_gac(
                init, iterations, smoothing, threshold, balloon
            )
        else:
            raise ValueError(
                f"Unknown method {method!r}. Use 'chan_vese', "
                "'morphological_chan_vese' or 'morphological_geodesic'."
            )

        self.mask_ = mask.astype(np.uint8)
        self._fitted = True
        self._method = method
        return self

    # ------------------------------------------------------------------ #
    #  2D slice-wise Chan-Vese                                            #
    # ------------------------------------------------------------------ #
    def _run_chan_vese_slice_wise(
        self,
        init: np.ndarray,
        iterations: int,
        dt: float,
        lambda1: float,
        lambda2: float,
    ) -> np.ndarray:
        from skimage.segmentation import chan_vese

        out = np.zeros_like(self.volume_, dtype=np.uint8)
        nz = self.volume_.shape[2]
        for z in range(nz):
            slc = self.volume_[:, :, z]
            init_s = init[:, :, z]
            if init_s.sum() == 0:
                # Auto-init: small circle in the centre.
                ny, nx = slc.shape
                init_s = np.zeros((ny, nx), dtype=np.float32)
                yy, xx = np.ogrid[:ny, :nx]
                r2 = (yy - ny // 2) ** 2 + (xx - nx // 2) ** 2
                init_s[r2 < (min(ny, nx) // 4) ** 2] = 1
            try:
                cv = chan_vese(
                    slc,
                    mu=0.1,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    tol=1e-4,
                    max_iter=iterations,
                    dt=dt,
                    init_level_set=init_s,
                )
                out[:, :, z] = (cv > 0).astype(np.uint8)
            except Exception:
                out[:, :, z] = init_s.astype(np.uint8)
        return out

    # ------------------------------------------------------------------ #
    #  3D morphological Chan-Vese (ACWE)                                  #
    # ------------------------------------------------------------------ #
    def _run_morphological_cv(
        self,
        init: np.ndarray,
        iterations: int,
        smoothing: int,
        lambda1: float,
        lambda2: float,
    ) -> np.ndarray:
        from skimage.segmentation import morphological_chan_vese

        result = morphological_chan_vese(
            self.volume_,
            num_iter=iterations,
            init_level_set=init,
            smoothing=smoothing,
            lambda1=lambda1,
            lambda2=lambda2,
        )
        return (result > 0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  3D morphological GAC                                               #
    # ------------------------------------------------------------------ #
    def _run_morphological_gac(
        self,
        init: np.ndarray,
        iterations: int,
        smoothing: int,
        threshold,
        balloon: float,
    ) -> np.ndarray:
        from scipy.ndimage import gaussian_gradient_magnitude
        from skimage import segmentation as skseg

        gmag = gaussian_gradient_magnitude(self.volume_, sigma=1.0)
        if gmag.max() > 0:
            gmag = gmag / gmag.max()
        g = 1.0 / (1.0 + gmag)

        thr = float(np.median(g)) if threshold == "auto" else float(threshold)

        try:
            result = skseg.morphological_geodesic_active_contour(
                g,
                num_iter=iterations,
                init_level_set=init,
                smoothing=smoothing,
                threshold=thr,
                balloon=balloon,
            )
            return (result > 0).astype(np.uint8)
        except Exception:
            return init.astype(np.uint8)
