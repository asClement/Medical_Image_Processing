# src/denoise/spatial/anisotropic.py
import numpy as np
import SimpleITK as sitk

from denoise.base import BaseMedicalDenoiser
from medio.nifti import MedicalImage3D


class AnisotropicDenoiser(BaseMedicalDenoiser):
    """
    Filtre de diffusion anisotrope de Curvature Anisotropic (Perona-Malik).
    Ultra performant pour éliminer le bruit IRM tout en préservant la netteté.
    """

    def __init__(self, n_iter: int = 5, time_step: float = 0.0625, conductance: float = 9.0):
        self.n_iter = n_iter
        self.time_step = time_step
        self.conductance = conductance

    def filter(
        self, image: MedicalImage3D | np.ndarray, mask: np.ndarray | None = None
    ) -> MedicalImage3D | np.ndarray:
        is_medical_obj = isinstance(image, MedicalImage3D)
        # Si l'image est un MedicalImage3D, extraire les données
        if is_medical_obj:
            data = image.data
            spacing = image.spacing
        else:
            if not isinstance(image, np.ndarray):
                raise TypeError("Type non supporté: " + str(type(image)))
            data = image
            spacing = (1.0, 1.0, 1.0)
        # Convertir l'image en SimpleITK
        sitk_image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
        sitk_image.SetSpacing(spacing)

        # Appliquer le filtre anisotrope
        filtered_sitk_image = sitk.CurvatureAnisotropicDiffusionImageFilter()
        filtered_sitk_image.SetNumberOfIterations(self.n_iter)
        filtered_sitk_image.SetTimeStep(self.time_step)
        filtered_sitk_image.SetConductanceParameter(self.conductance)
        filtered_sitk_image = filtered_sitk_image.Execute(sitk_image)

        # Convertir l'image filtrée en numpy
        filtered_image = sitk.GetArrayFromImage(filtered_sitk_image)
        filtered_image = np.transpose(filtered_image, (2, 1, 0))

        # Retourner l'image filtrée
        if is_medical_obj:
            return MedicalImage3D(
                data=filtered_image.astype(np.float32),
                affine=image.affine,
                header=image.header,
                mask=image.mask,
                mask_header=image.mask_header,
            )
        return filtered_image.astype(np.float32)
