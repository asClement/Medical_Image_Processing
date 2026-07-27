# src/edges/classic/sobel.py
import numpy as np
from scipy.ndimage import sobel

from edges.base import BaseEdgeDetector
from medio.nifti import MedicalImage3D


class SobelEdgeDetector(BaseEdgeDetector):
    """Détecteur de contours Sobel (supporte nativement la 2D et la 3D)."""

    def __init__(self, apply_mask: bool = False):
        self.apply_mask = apply_mask

    def detect(
        self,
        image: MedicalImage3D | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> MedicalImage3D | np.ndarray:

        is_medical_obj = isinstance(image, MedicalImage3D)
        data = image.data if is_medical_obj else image

        # Normalisation préalable si besoin (type float pour la précision du gradient)
        data_float = data.astype(np.float32)

        # Calcul du gradient 3D (ou 2D selon ndim)
        # scipy.ndimage.sobel calcule le gradient axe par axe
        grad_components = [sobel(data_float, axis=i) for i in range(data_float.ndim)]

        # Magnitude du gradient : sqrt(Gx^2 + Gy^2 + Gz^2)
        magnitude = np.sqrt(sum(g**2 for g in grad_components))

        # Normalisation du résultat sur [0, 255]
        if magnitude.max() > 0:
            magnitude = (magnitude / magnitude.max()) * 255.0
        magnitude = magnitude.astype(np.uint8)

        # Application éventuelle du masque ROI
        if mask is not None and self.apply_mask:
            magnitude[mask == 0] = 0

        # Output
        if is_medical_obj:
            return MedicalImage3D(data=magnitude, affine=image.affine, header=image.header)

        return magnitude
