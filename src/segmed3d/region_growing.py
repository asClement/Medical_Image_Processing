"""Region-growing segmentation with 26-connectivity.

Two modes are provided:

- ``mode='range'``: include voxels whose intensity is within
  ``[seed_value - tolerance, seed_value + tolerance]`` and connected to the
  seed.
- ``mode='gradient'``: include voxels whose intensity differs from a
  neighbour already in the region by at most ``tolerance`` (more robust to
  smooth intensity ramps).
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._base import BaseSegmentation

__all__ = ["RegionGrowingSegmentation"]


class RegionGrowingSegmentation(BaseSegmentation):
    """3D region growing with 6/18/26 connectivity.

    Examples
    --------
    >>> seg = RegionGrowingSegmentation(volume)
    >>> seg.fit(seed_point=(45, 50, 32), tolerance=30, connectivity=26)
    >>> mask = seg.get_mask()
    """

    def fit(
        self,
        seed_point: Union[Sequence[int], np.ndarray],
        tolerance: Optional[float] = None,
        connectivity: int = 26,
        mode: str = "range",
        max_size: Optional[int] = None,
    ) -> "RegionGrowingSegmentation":
        """Grow the region from ``seed_point``.

        Parameters
        ----------
        seed_point : tuple of int
            ``(x, y, z)`` voxel coordinates of the seed.
        tolerance : float, optional
            Intensity tolerance.  Defaults to ``0.1 * (vol.max() - vol.min())``.
        connectivity : int, {6, 18, 26}
            Voxel neighbourhood.
        mode : str, {'range', 'gradient'}
            - ``'range'``: absolute tolerance around the seed value.
            - ``'gradient'``: relative tolerance between adjacent voxels.
        max_size : int, optional
            Maximum number of voxels in the region (safety cap).

        Returns
        -------
        self
        """
        from scipy.ndimage import label

        seed = tuple(int(s) for s in seed_point)
        if len(seed) != 3:
            raise ValueError("seed_point must be (x, y, z).")
        for i, s in enumerate(seed):
            if not 0 <= s < self.volume_.shape[i]:
                raise ValueError(
                    f"Seed coordinate {s} is out of bounds along axis {i} "
                    f"(shape {self.volume_.shape})."
                )

        vol = self.volume_
        seed_value = float(vol[seed])
        if tolerance is None:
            tolerance = 0.1 * float(vol.max() - vol.min()) if vol.max() > vol.min() else 1.0

        # Build candidate mask
        if mode == "range":
            candidate = (np.abs(vol - seed_value) <= tolerance)
        elif mode == "gradient":
            candidate = self._gradient_candidate(vol, seed, tolerance)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'range' or 'gradient'.")

        # Connectivity structure
        if connectivity == 6:
            struct = np.array([
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            ])
        elif connectivity == 18:
            struct = np.array([
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
                [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            ])
        elif connectivity == 26:
            struct = np.ones((3, 3, 3), dtype=np.uint8)
        else:
            raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}.")

        # Find the connected component containing the seed among candidates.
        lbl, _ = label(candidate, structure=struct)
        seed_label = lbl[seed]
        if seed_label == 0:
            # Seed itself failed the candidate criterion — start a tiny
            # region from the seed alone.
            mask = np.zeros_like(vol, dtype=np.uint8)
            mask[seed] = 1
        else:
            mask = (lbl == seed_label).astype(np.uint8)

        if max_size is not None and int(mask.sum()) > max_size:
            # Trim by keeping only the closest voxels (by intensity distance).
            dist = np.abs(vol - seed_value)
            coords = np.argwhere(mask)
            dists = dist[coords[:, 0], coords[:, 1], coords[:, 2]]
            order = np.argsort(dists)[:max_size]
            keep = coords[order]
            mask = np.zeros_like(vol, dtype=np.uint8)
            mask[keep[:, 0], keep[:, 1], keep[:, 2]] = 1

        self.mask_ = mask
        self._seed_value_ = seed_value
        self._seed_point_ = seed
        self._tolerance_ = float(tolerance)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    #  Gradient-mode candidate mask                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gradient_candidate(vol: np.ndarray, seed: tuple, tol: float) -> np.ndarray:
        """BFS region growing with per-edge tolerance."""
        from collections import deque

        shape = vol.shape
        visited = np.zeros(shape, dtype=bool)
        sx, sy, sz = seed
        visited[sx, sy, sz] = True
        queue = deque([seed])
        seed_val = float(vol[seed])

        # 26-connectivity offsets
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    offsets.append((dx, dy, dz))

        while queue:
            x, y, z = queue.popleft()
            cur_val = float(vol[x, y, z])
            for dx, dy, dz in offsets:
                nx, ny, nz = x + dx, y + dy, z + dz
                if not (0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]):
                    continue
                if visited[nx, ny, nz]:
                    continue
                neigh_val = float(vol[nx, ny, nz])
                if abs(neigh_val - cur_val) <= tol:
                    visited[nx, ny, nz] = True
                    queue.append((nx, ny, nz))

        return visited

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    def get_seed_value(self) -> float:
        """Return the intensity at the seed point."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._seed_value_

    def get_seed_point(self) -> tuple:
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._seed_point_
