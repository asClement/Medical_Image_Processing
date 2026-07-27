# src/edges/classic/canny.py
import numpy as np
from skimage.feature import canny

from edges.base import BaseEdgeDetector
from medio.nifti import MedicalImage3D


class CannyEdgeDetector(BaseEdgeDetector):
    """Détecteur de contours Canny pour les images médicales 3D."""

    def __init__(self, sigma=1.0, low_threshold=None, high_threshold=None):
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def detect(
        self,
        image: MedicalImage3D | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> MedicalImage3D | None:
        """
        Détecte les contours dans l'image en utilisant l'algorithme Canny.
        """
        if isinstance(image, MedicalImage3D):
            data = image.data
        else:
            data = image
        edges = canny(
            data,
            sigma=self.sigma,
            low_threshold=self.low_threshold,
            high_threshold=self.high_threshold,
        )

        return MedicalImage3D(edges, affine=image.affine, header=image.header)
