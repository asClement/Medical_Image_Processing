from pathlib import Path

import matplotlib.pyplot as plt

from denoise.anisotropic import AnisotropicDenoiser
from denoise.nlmeans import NLMRicianDenoiser
from edges.classic.canny import CannyEdgeDetector
from medio.nifti import load_nifti

# 1. Charger l'IRM DCE
PATIENT_ID = "sub-ISPY2-125130"
SESSION_ID = "ses-T0"
SERIES_ID = "dce-post-1"
NIFTI_FOLDER = "ISPY2_dataset/derivatives/nifti"
nifti_file = (
    Path(NIFTI_FOLDER)
    / f"{PATIENT_ID}/{SESSION_ID}/perf/{PATIENT_ID}_{SESSION_ID}_{SERIES_ID}.nii.gz"
)
mask_file = (
    Path(NIFTI_FOLDER)
    / f"{PATIENT_ID}/{SESSION_ID}/seg/{PATIENT_ID}_{SESSION_ID}_mask.nii.gz"
)
mask = load_nifti(mask_file).data
irm = load_nifti(nifti_file, mask_file)

# 2. Débruiter pour les analyses de radiomique
denoiser_rician = NLMRicianDenoiser()
denoiser_anisotropic = AnisotropicDenoiser()

irm_denoised = denoiser_rician.filter(irm)
irm_denoised = denoiser_anisotropic.filter(irm_denoised)

# Détection des contours
canny = CannyEdgeDetector()
edges = canny.detect(irm_denoised, mask)

# Affichage
slice_index = irm.data.shape[2] // 2
slice_original = irm.data[:, :, slice_index]
slice_edges = edges.data[:, :, slice_index]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(slice_original.T, cmap="gray", origin="lower")
axes[0].set_title("Coupe IRM originale")

axes[1].imshow(slice_edges.T, cmap="gray", origin="lower")
axes[1].set_title("Contours (Canny)")
plt.show()
