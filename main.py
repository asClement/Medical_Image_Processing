from pathlib import Path

from denoise.io.nifti import load_nifti
from denoise.spatial.anisotropic import AnisotropicDenoiser
from denoise.spatial.gaussian import GaussianDenoiser
from denoise.statistical.nlmeans import NLMRicianDenoiser

# 1. Charger l'IRM DCE et le masque de la tumeur ISPY2
PATIENT_ID = "sub-ISPY2-125130"
SESSION_ID = "ses-T0"
SERIES_ID = "dce-post-1"
NIFTI_FOLDER = "ISPY2_dataset/derivatives/nifti"
nifti_file = Path(NIFTI_FOLDER) / f"{PATIENT_ID}/{SESSION_ID}/perf/{PATIENT_ID}_{SESSION_ID}_{SERIES_ID}.nii.gz"
irm = load_nifti(nifti_file)

# 2. Débruiter uniquement la tumeur pour les analyses de radiomique
denoiser = GaussianDenoiser()
denoiser_rician = NLMRicianDenoiser()
denoiser_anisotropic = AnisotropicDenoiser()
irm_denoised = denoiser_rician.filter(irm)
irm_denoised = denoiser_anisotropic.filter(irm_denoised)

denoisers = [denoiser, denoiser_rician, denoiser_anisotropic]
# tester les différents débruitages
for denoiser in denoisers:
    irm_denoised = denoiser.filter(irm_denoised)
