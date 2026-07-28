"""Clustering-based segmentation: K-Means and Fuzzy C-Means.

Both algorithms operate on an augmented feature space combining intensity
and (optionally) spatial coordinates, producing a 3D label volume.  The
foreground cluster is selected as the one with the highest mean intensity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._base import BaseSegmentation

__all__ = ["ClusteringSegmentation"]


class ClusteringSegmentation(BaseSegmentation):
    """3D clustering segmentation.

    Examples
    --------
    >>> seg = ClusteringSegmentation(volume)
    >>> seg.fit(method='kmeans', n_clusters=3, spatial_weight=0.3)
    >>> mask = seg.get_mask()
    """

    def fit(
        self,
        method: str = "kmeans",
        n_clusters: int = 3,
        spatial_weight: float = 0.5,
        random_state: int = 42,
        fuzzy_m: float = 2.0,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> "ClusteringSegmentation":
        """Run the clustering.

        Parameters
        ----------
        method : str, {'kmeans', 'fcm'}
            - ``'kmeans'``: scikit-learn K-Means.
            - ``'fcm'``: Fuzzy C-Means (requires the optional ``scikit-fuzzy``
              dependency).
        n_clusters : int
            Number of clusters (>=2).
        spatial_weight : float
            Weight of the normalised spatial coordinates in the feature
            vector.  ``0.0`` = intensity-only, ``1.0`` = equal weighting.
        random_state : int
            RNG seed for reproducibility.
        fuzzy_m : float
            Fuzziness exponent for FCM (>1, typically 2).
        max_iter : int
            Maximum number of iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        self
        """
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")
        if not 0.0 <= spatial_weight <= 1.0:
            raise ValueError("spatial_weight must be in [0, 1].")

        X, shape = self._build_features(spatial_weight)

        if method == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=max_iter,
                tol=tol,
            )
            labels = km.fit_predict(X)
            self._cluster_volume_ = labels.reshape(shape).astype(np.int32)
            self._membership_volume_ = None

        elif method == "fcm":
            try:
                import skfuzzy as fuzz
            except ImportError as exc:
                raise ImportError(
                    "Fuzzy C-Means requires the optional dependency "
                    "'scikit-fuzzy'. Install it with `pip install segmed3d[fuzzy]`."
                ) from exc
            # skfuzzy expects features in shape (n_features, n_samples)
            cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
                X.T,
                c=n_clusters,
                m=fuzzy_m,
                error=tol,
                maxiter=max_iter,
                seed=random_state,
                init=None,
            )
            labels = np.argmax(u, axis=0)
            self._cluster_volume_ = labels.reshape(shape).astype(np.int32)
            self._membership_volume_ = u.T.reshape((*shape, n_clusters)).astype(np.float32)

        else:
            raise ValueError(f"Unknown method {method!r}. Use 'kmeans' or 'fcm'.")

        # Pick the foreground cluster: highest mean intensity.
        vol = self.volume_
        flat_vol = vol.ravel()
        means = []
        for k in range(n_clusters):
            mask_k = (self._cluster_volume_.ravel() == k)
            if mask_k.sum() == 0:
                means.append(-np.inf)
            else:
                means.append(float(flat_vol[mask_k].mean()))
        fg_label = int(np.argmax(means))
        self.mask_ = (self._cluster_volume_ == fg_label).astype(np.uint8)
        self._foreground_label_ = fg_label
        self._cluster_means_ = means
        self._fitted = True
        self._method = method
        return self

    # ------------------------------------------------------------------ #
    #  Feature builder                                                    #
    # ------------------------------------------------------------------ #
    def _build_features(self, spatial_weight: float):
        """Build the (n_voxels, n_features) feature matrix.

        Features:
        - Intensity (normalised to [0,1]).
        - (1 - spatial_weight) weight on intensity, spatial_weight on
          normalised (x, y, z) coordinates.
        """
        vol = self.volume_
        shape = vol.shape
        # Intensity feature
        v = vol.ravel().astype(np.float32)
        v_min, v_max = float(v.min()), float(v.max())
        if v_max - v_min > 1e-8:
            v_norm = (v - v_min) / (v_max - v_min)
        else:
            v_norm = np.zeros_like(v)
        intensity_weight = 1.0 - spatial_weight

        if spatial_weight == 0.0:
            X = v_norm.reshape(-1, 1) * intensity_weight
        else:
            # Spatial coordinates normalised to [0,1]
            xs = np.linspace(0, 1, shape[0], dtype=np.float32)
            ys = np.linspace(0, 1, shape[1], dtype=np.float32)
            zs = np.linspace(0, 1, shape[2], dtype=np.float32)
            xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
            xx = xx.ravel()
            yy = yy.ravel()
            zz = zz.ravel()
            X = np.stack([
                v_norm * intensity_weight,
                xx * spatial_weight,
                yy * spatial_weight,
                zz * spatial_weight,
            ], axis=1).astype(np.float32)
        return X, shape

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    def get_cluster_volume(self) -> np.ndarray:
        """Return the full integer label volume (one int per cluster)."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._cluster_volume_.copy()

    def get_membership_volume(self) -> Optional[np.ndarray]:
        """Return the fuzzy membership volume (FCM only) or ``None``."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        if self._membership_volume_ is None:
            return None
        return self._membership_volume_.copy()

    def get_foreground_label(self) -> int:
        """Return the cluster label selected as foreground."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._foreground_label_
