"""segmed3d — Mathematical morphology & segmentation for 3D medical images.

A reusable Python library implementing 7 classical 3D segmentation methods
with a uniform scikit-learn-style API:

    seg = XxxSegmentation(volume, affine, header)
    seg.fit(...)
    mask = seg.get_mask()
    seg.save('mask.nii.gz')
    # Or, in one line:
    mask = XxxSegmentation(volume, affine, header)(...)

Available segmentation classes
------------------------------
- :class:`ThresholdSegmentation`       — Otsu, multi-Otsu, slice-wise Otsu
- :class:`WatershedSegmentation`        — gradient / distance / image watershed
- :class:`ActiveContourSegmentation`    — slice-wise snakes + 3D morphological GAC
- :class:`LevelSetSegmentation`         — Chan-Vese (2D slice-wise + 3D morphological)
- :class:`RegionGrowingSegmentation`    — 6/18/26-connected region growing
- :class:`ClusteringSegmentation`       — K-Means + Fuzzy C-Means
- :class:`AtlasSegmentation`            — multi-atlas with SimpleITK registration + 4 fusions

Utility classes (in :mod:`segmed3d.utils`)
------------------------------------------
- :class:`ImageIO`        — NIfTI load/save
- :class:`Preprocessor`   — normalise, denoise, N4 bias correction
- :class:`Postprocessor`  — mask cleaning, morphology
- :class:`Metrics`        — Dice / IoU / Hausdorff95 / ...
- :class:`Visualizer`     — slice / overlay / surface plots

Quick example
-------------
>>> from segmed3d import (
...     ImageIO, Preprocessor, ThresholdSegmentation,
...     Postprocessor, Visualizer, Metrics,
... )
>>> vol, affine, hdr = ImageIO.load_nifti('tumor.nii.gz')
>>> vol = Preprocessor.normalize(Preprocessor.bias_field_correction(vol))
>>> mask = ThresholdSegmentation(vol, affine, hdr)(method='otsu')
>>> mask = Postprocessor.clean_mask(mask, min_size=50)
>>> Visualizer.plot_3d_slices(vol, mask)
>>> gt, _, _ = ImageIO.load_mask('gt.nii.gz')
>>> print(Metrics.all_metrics(mask, gt))
"""

from ._version import __version__
from ._base import BaseSegmentation
from .threshold import ThresholdSegmentation
from .watershed import WatershedSegmentation
from .active_contour import ActiveContourSegmentation
from .level_set import LevelSetSegmentation
from .region_growing import RegionGrowingSegmentation
from .clustering import ClusteringSegmentation
from .atlas import AtlasSegmentation
from .utils import ImageIO, Preprocessor, Postprocessor, Metrics, Visualizer

__all__ = [
    "__version__",
    # Base
    "BaseSegmentation",
    # Segmentation algorithms
    "ThresholdSegmentation",
    "WatershedSegmentation",
    "ActiveContourSegmentation",
    "LevelSetSegmentation",
    "RegionGrowingSegmentation",
    "ClusteringSegmentation",
    "AtlasSegmentation",
    # Utilities
    "ImageIO",
    "Preprocessor",
    "Postprocessor",
    "Metrics",
    "Visualizer",
]
