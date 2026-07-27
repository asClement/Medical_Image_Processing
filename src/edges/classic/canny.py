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
    ) -> MedicalImage3D | np.ndarray:
        """
        Détecte les contours dans l'image en utilisant l'algorithme Canny.
        """
        # extraction des données
        is_medical_obj = isinstance(image, MedicalImage3D)
        data = image.data if is_medical_obj else image

        if data.ndim == 2:
            edges = canny(
                data,
                sigma=self.sigma,
                low_threshold=self.low_threshold,
                high_threshold=self.high_threshold,
                mask=mask,
            )
        # si c'est une image 3D, appliquer Canny à chaque slice
        elif data.ndim == 3:
            edges = np.zeros_like(data, dtype=bool)

            for z in range(data.shape[2]):
                slice_mask = mask[:, :, z] if mask is not None else None
                edges[:, :, z] = canny(
                    data[:, :, z],
                    sigma=self.sigma,
                    low_threshold=self.low_threshold,
                    high_threshold=self.high_threshold,
                    mask=slice_mask,
                )
        else:
            raise ValueError("Dimension non supportée")

        # application du masque sur le total
        if mask is not None:
            edges = edges * mask

        # conversion en MedicalImage3D
        if is_medical_obj:
            return MedicalImage3D(edges, affine=image.affine, header=image.header)

        return edges.astype(np.uint8)
