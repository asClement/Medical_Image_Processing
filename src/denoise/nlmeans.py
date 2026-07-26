# src/denoise/statistical/nlm_rician.py
import numpy as np
from dipy.denoise.nlmeans import nlmeans
from dipy.denoise.noise_estimate import estimate_sigma

from denoise.base import BaseMedicalDenoiser
from medio.nifti import MedicalImage3D


class NLMRicianDenoiser(BaseMedicalDenoiser):
    """
    Filtre Non-Local Means (NLM) adapté au bruit de Rician en IRM.
    Utilise DIPY sous le capot pour le calcul 3D.
    Avec les masques de segmentation pour améliorer la précision sur les zones médicales
    """

    def __init__(self, sigma: None | float = 1.0, rician: bool = False, use_mask: bool = False):
        """
        :param sigma: Niveau de bruit estimé. Si None, il sera estimé automatiquement.
        :param rician: Si True, applique le biais de correction Rician (spécifique IRM).
        :param use_mask: Si True, applique le masque de segmentation avant le filtrage.
        """
        self.sigma = sigma
        self.rician = rician
        self.use_mask = use_mask

    def filter(
        self, image: MedicalImage3D | np.ndarray, mask: np.ndarray | None = None
    ) -> MedicalImage3D | np.ndarray:
        # extraire l'array numpy si c'est un objet MedicalImage3D
        if isinstance(image, MedicalImage3D):
            data = image.data
            affine = image.affine
            header = image.header
            effective_mask = mask if mask is not None else image.mask
            is_medical_obj = True
        else:
            data = image
            effective_mask = mask
            is_medical_obj = False

        # estimation automatique du bruit
        if self.sigma is None:
            noise_sigma = estimate_sigma(data)
        else:
            noise_sigma = self.sigma

        # preparation du masque
        mask_to_pass = effective_mask if (self.use_mask and effective_mask is not None) else None

        # application du filtre NLM
        denoised = nlmeans(
            data,
            sigma=noise_sigma,
            mask=mask_to_pass,
            rician=self.rician,
            method="blockwise",
        )

        # converstion en float32
        denoised_data = denoised.astype(np.float32)

        if is_medical_obj:
            return MedicalImage3D(
                data=denoised_data, affine=affine, header=header, mask=effective_mask
            )
        return denoised_data
