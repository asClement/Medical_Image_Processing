"""Evaluation metrics for 3D binary segmentation.

All metrics assume binary masks (values in {0, 1}) of identical shape.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["Metrics"]


class Metrics:
    """Static collection of segmentation evaluation metrics."""

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _check(pred: np.ndarray, gt: np.ndarray) -> None:
        if pred.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: pred {pred.shape} vs gt {gt.shape}."
            )

    @staticmethod
    def _binarise(arr: np.ndarray) -> np.ndarray:
        return (np.asarray(arr) > 0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Overlap metrics                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def dice(pred: np.ndarray, gt: np.ndarray) -> float:
        """Dice similarity coefficient ``2|A∩B| / (|A|+|B|)``."""
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        sp, sg = p.sum(), g.sum()
        if sp == 0 and sg == 0:
            return 1.0
        inter = float(np.logical_and(p, g).sum())
        denom = float(sp + sg)
        return (2.0 * inter) / denom if denom > 0 else 1.0

    @staticmethod
    def iou(pred: np.ndarray, gt: np.ndarray) -> float:
        """Intersection-over-Union (Jaccard index)."""
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        union = float(np.logical_or(p, g).sum())
        if union == 0:
            return 1.0
        inter = float(np.logical_and(p, g).sum())
        return inter / union

    @staticmethod
    def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Sensitivity (recall / TPR): ``TP / (TP + FN)``."""
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        tp = float(np.logical_and(p == 1, g == 1).sum())
        fn = float(np.logical_and(p == 0, g == 1).sum())
        return tp / (tp + fn) if (tp + fn) > 0 else 1.0

    @staticmethod
    def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
        """Specificity (TNR): ``TN / (TN + FP)``."""
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        tn = float(np.logical_and(p == 0, g == 0).sum())
        fp = float(np.logical_and(p == 1, g == 0).sum())
        return tn / (tn + fp) if (tn + fp) > 0 else 1.0

    @staticmethod
    def precision(pred: np.ndarray, gt: np.ndarray) -> float:
        """Precision (PPV): ``TP / (TP + FP)``."""
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        tp = float(np.logical_and(p == 1, g == 1).sum())
        fp = float(np.logical_and(p == 1, g == 0).sum())
        return tp / (tp + fp) if (tp + fp) > 0 else 1.0

    # ------------------------------------------------------------------ #
    #  Surface metrics                                                    #
    # ------------------------------------------------------------------ #
    @staticmethod
    def hausdorff95(
        pred: np.ndarray,
        gt: np.ndarray,
        voxel_spacing: Optional[tuple] = None,
    ) -> float:
        """95th-percentile symmetric Hausdorff distance (in mm).

        Uses :func:`medpy.metric.binary.hd95` if available, otherwise falls
        back to a pure-scipy implementation.
        """
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)

        try:
            from medpy.metric.binary import hd95
            return float(hd95(p, g, voxelspacing=voxel_spacing))
        except Exception:
            return Metrics._hd95_scipy(p, g, voxel_spacing)

    @staticmethod
    def _hd95_scipy(p: np.ndarray, g: np.ndarray, spacing: tuple) -> float:
        """Fallback HD95 implementation using scipy cKDTree."""
        from scipy.spatial import cKDTree
        from scipy.ndimage import binary_erosion

        struct = np.ones((3, 3, 3), dtype=np.uint8)

        def surface(mask: np.ndarray) -> np.ndarray:
            if mask.sum() == 0:
                return np.empty((0, 3), dtype=np.float32)
            eroded = binary_erosion(mask, structure=struct)
            surf = mask & ~eroded
            idx = np.argwhere(surf).astype(np.float32)
            idx *= np.asarray(spacing, dtype=np.float32)
            return idx

        sp, sg = surface(p), surface(g)
        if sp.shape[0] == 0 or sg.shape[0] == 0:
            return float("inf")
        tree_g = cKDTree(sg)
        d_pg, _ = tree_g.query(sp, k=1)
        tree_p = cKDTree(sp)
        d_gp, _ = tree_p.query(sg, k=1)
        all_d = np.concatenate([d_pg, d_gp])
        return float(np.percentile(all_d, 95))

    # ------------------------------------------------------------------ #
    #  Volume metric                                                      #
    # ------------------------------------------------------------------ #
    @staticmethod
    def volume_similarity(
        pred: np.ndarray,
        gt: np.ndarray,
        voxel_spacing: Optional[tuple] = None,
    ) -> float:
        """Relative volume difference ``1 - |Vp - Vg| / max(Vp, Vg)``.

        Values are in ``[0, 1]`` (1 = identical volumes).
        """
        p = Metrics._binarise(pred)
        g = Metrics._binarise(gt)
        Metrics._check(p, g)
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
        v_voxel = float(np.prod(voxel_spacing))
        vp = float(p.sum()) * v_voxel
        vg = float(g.sum()) * v_voxel
        if max(vp, vg) == 0:
            return 1.0
        return 1.0 - abs(vp - vg) / max(vp, vg)

    # ------------------------------------------------------------------ #
    #  All-in-one                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def all_metrics(
        pred: np.ndarray,
        gt: np.ndarray,
        voxel_spacing: Optional[tuple] = None,
    ) -> Dict[str, float]:
        """Compute every available metric in one call.

        Returns
        -------
        dict
            Keys: ``'dice'``, ``'iou'``, ``'sensitivity'``, ``'specificity'``,
            ``'precision'``, ``'hausdorff95'``, ``'volume_similarity'``.
        """
        if voxel_spacing is None:
            voxel_spacing = (1.0, 1.0, 1.0)
        return {
            "dice": Metrics.dice(pred, gt),
            "iou": Metrics.iou(pred, gt),
            "sensitivity": Metrics.sensitivity(pred, gt),
            "specificity": Metrics.specificity(pred, gt),
            "precision": Metrics.precision(pred, gt),
            "hausdorff95": Metrics.hausdorff95(pred, gt, voxel_spacing),
            "volume_similarity": Metrics.volume_similarity(pred, gt, voxel_spacing),
        }
