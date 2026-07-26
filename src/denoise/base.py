# src/denoise/base.py
from abc import ABC, abstractmethod

import numpy as np

from medio.nifti import MedicalImage3D


class BaseMedicalDenoiser(ABC):
    """
    Classe abstraite de base pour les algorithmes de débruitage médical.
    """

    @abstractmethod
    def filter(
        self,
        image: MedicalImage3D | np.ndarray,
        mask: np.ndarray | None = None,
    ) -> MedicalImage3D | np.ndarray:
        """
        Applique le filtre de débruitage sur le volume IRM.

        Parameters
        ----------
        image : Union[MedicalImage3D, np.ndarray]
            Le volume IRM à débruité.

        Returns
        -------
        Union[MedicalImage3D, np.ndarray]
            Le volume 3D filtré de même structure et mêmes dimensions.
        """
        pass

    def __repr__(self) -> str:
        """
        Affiche le nom du filtre et ses paramètres.
        """
        params = ", ".join([f"{k}={v}" for k, v in self.__dict__.items()])
        return f"{self.__class__.__name__}({params})"
