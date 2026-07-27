from abc import ABC, abstractmethod

import numpy as np

from medio.nifti import MedicalImage3D


class BaseEdgeDetector(ABC):
    """Interface de base pour les detecteurs de contours"""

    @abstractmethod
    def detect(
        self,
        image: MedicalImage3D | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> MedicalImage3D | np.ndarray:
        """
        Détecte les contours dans un image médicale 3D ou un array numpy."""
        pass
