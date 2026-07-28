"""Utility subpackage for :mod:`segmed3d`.

Re-exports the public utility classes:

- :class:`ImageIO`         — NIfTI I/O
- :class:`Preprocessor`    — intensity normalisation, denoising, N4 bias correction
- :class:`Postprocessor`   — mask cleaning, hole filling, morphology
- :class:`Metrics`         — Dice / IoU / Hausdorff95 / etc.
- :class:`Visualizer`      — slice / overlay / 3D surface plots
"""

from .io import ImageIO
from .preprocessing import Preprocessor
from .postprocessing import Postprocessor
from .metrics import Metrics
from .visualization import Visualizer

__all__ = [
    "ImageIO",
    "Preprocessor",
    "Postprocessor",
    "Metrics",
    "Visualizer",
]
