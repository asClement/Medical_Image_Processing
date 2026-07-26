# src/denoise/spatial/gaussian.py
import numpy as np
from scipy.ndimage import gaussian_filter

from denoise.base import BaseMedicalDenoiser
from medio.nifti import MedicalImage3D


class GaussianDenoiser(BaseMedicalDenoiser):
    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma

    def filter(
        self, image: MedicalImage3D | np.ndarray, mask: np.ndarray | None = None
    ) -> MedicalImage3D | np.ndarray:

        # traitement si c'est un objet MedicalImage3D
        if isinstance(image, MedicalImage3D):
            filtered_data = gaussian_filter(image.data, sigma=self.sigma)
            return MedicalImage3D(data=filtered_data, affine=image.affine, header=image.header)
        # traitement si c'est un np.ndarray
        elif isinstance(image, np.ndarray):
            return gaussian_filter(image, sigma=self.sigma)
        else:
            raise TypeError("Type non supporté: " + str(type(image)))
