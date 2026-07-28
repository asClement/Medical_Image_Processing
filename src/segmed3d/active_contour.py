"""Active-contour segmentation for 3D volumes.

Because :func:`skimage.segmentation.active_contour` is 2D-only, this module
provides a dual approach:

- ``method='slice_wise'``: classic parametric snakes applied independently on
  each Z-slice, then stacked back into a 3D mask.  Best for elongated
  structures aligned with the Z axis.
- ``method='morphological_geodesic'``: true 3D segmentation using
  :func:`skimage.segmentation.morphological_geodesic_active_contour`, which
  evolves a level-set-like contour guided by an edge indicator.

Both methods require an initial mask (``init_mask``) close to the target.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseSegmentation

__all__ = ["ActiveContourSegmentation"]


class ActiveContourSegmentation(BaseSegmentation):
    """3D active-contour segmentation (parametric snakes + morphological GAC).

    Examples
    --------
    >>> from segmed3d import ThresholdSegmentation
    >>> init = ThresholdSegmentation(vol)(method='otsu')  # rough init
    >>> seg = ActiveContourSegmentation(vol)
    >>> seg.fit(init_mask=init, method='morphological_geodesic')
    >>> mask = seg.get_mask()
    """

    def fit(
        self,
        init_mask: np.ndarray,
        method: str = "morphological_geodesic",
        alpha: float = 0.01,
        beta: float = 0.1,
        gamma: float = 0.01,
        w_line: float = 0.0,
        w_edge: float = 1.0,
        max_iter: int = 2500,
        iterations: int = 50,
        smoothing: int = 1,
        threshold: str = "auto",
        balloon: float = 0.0,
        sigma: float = 1.0,
    ) -> "ActiveContourSegmentation":
        """Run the active contour.

        Parameters
        ----------
        init_mask : np.ndarray
            Initial binary mask (same shape as the volume).
        method : str, {'slice_wise', 'morphological_geodesic'}
            Strategy (see module docstring).
        alpha, beta, gamma : float
            Snakes elasticity / rigidity / viscosity (slice_wise only).
        w_line, w_edge : float
            Weights of the line / edge potential (slice_wise only).
        max_iter : int
            Maximum snake iterations per slice (slice_wise only).
        iterations : int
            Number of morphological GAC iterations.
        smoothing : int
            Number of smoothing repetitions per GAC iteration.
        threshold : str or float
            Edge threshold; ``'auto'`` uses the median of the gradient
            magnitude.
        balloon : float
            Balloon force (positive inflates the contour).
        sigma : float
            Gaussian sigma applied before gradient computation.

        Returns
        -------
        self
        """
        init = (np.asarray(init_mask) > 0).astype(np.float32)
        if init.shape != self.volume_.shape:
            raise ValueError(
                f"init_mask shape {init.shape} != volume {self.volume_.shape}."
            )

        if method == "slice_wise":
            mask = self._run_slice_wise(
                init, alpha, beta, gamma, w_line, w_edge, max_iter
            )
        elif method == "morphological_geodesic":
            mask = self._run_morphological_gac(
                init, iterations, smoothing, threshold, balloon, sigma
            )
        else:
            raise ValueError(
                f"Unknown method {method!r}. "
                "Use 'slice_wise' or 'morphological_geodesic'."
            )

        self.mask_ = mask.astype(np.uint8)
        self._fitted = True
        self._method = method
        return self

    # ------------------------------------------------------------------ #
    #  Slice-wise classic snakes                                          #
    # ------------------------------------------------------------------ #
    def _run_slice_wise(
        self,
        init: np.ndarray,
        alpha: float,
        beta: float,
        gamma: float,
        w_line: float,
        w_edge: float,
        max_iter: int,
    ) -> np.ndarray:
        from skimage.filters import gaussian
        from skimage.segmentation import active_contour

        out = np.zeros_like(self.volume_, dtype=np.uint8)
        nz = self.volume_.shape[2]
        for z in range(nz):
            slc = self.volume_[:, :, z]
            init_s = init[:, :, z]
            if init_s.sum() == 0:
                continue
            try:
                from skimage.measure import find_contours
                contours = find_contours(init_s, level=0.5)
                if not contours:
                    continue
                snake = max(contours, key=len)
                if len(snake) < 4:
                    continue
                smoothed = gaussian(slc, sigma=1.0, preserve_range=True)
                snake = active_contour(
                    smoothed,
                    snake,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    w_line=w_line,
                    w_edge=w_edge,
                    max_num_iter=max_iter,
                )
                # Rasterise the snake back to a mask.
                from skimage.draw import polygon
                rr, cc = polygon(snake[:, 0], snake[:, 1], slc.shape)
                slice_mask = np.zeros_like(slc, dtype=np.uint8)
                slice_mask[rr, cc] = 1
                out[:, :, z] = slice_mask
            except Exception:
                # Fallback: keep the init for this slice.
                out[:, :, z] = init_s.astype(np.uint8)
        return out

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
        sigma: float,
    ) -> np.ndarray:
        from skimage import segmentation as skseg

        # Edge indicator: g = 1 / (1 + |∇Gσ * I|)
        from scipy.ndimage import gaussian_gradient_magnitude
        smoothed = gaussian_gradient_magnitude(self.volume_, sigma=sigma)
        # Normalise
        if smoothed.max() > 0:
            smoothed = smoothed / smoothed.max()
        g = 1.0 / (1.0 + smoothed)

        if threshold == "auto":
            thr = float(np.median(g))
        else:
            thr = float(threshold)

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
