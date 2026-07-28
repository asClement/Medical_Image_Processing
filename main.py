from pathlib import Path

import matplotlib.pyplot as plt

from denoise import AnisotropicDenoiser, NLMRicianDenoiser
from edges import CannyEdgeDetector, SobelEdgeDetector
from mathmorpho import (
    MathMorphology,
    MorphologyCleaning,
    MorphologyEnhancement,
    MorphologyStats,
)
from medio.nifti import load_nifti
from preprocessing import Preprocessing
from segmed3d import Metrics, Postprocessor, ThresholdSegmentation

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
    Path(NIFTI_FOLDER) / f"{PATIENT_ID}/{SESSION_ID}/seg/{PATIENT_ID}_{SESSION_ID}_mask.nii.gz"
)
mask = load_nifti(mask_file).data
irm = load_nifti(nifti_file, mask_file)

# 2. Prétraitement d'intensité (preprocessing.py)
preproc = Preprocessing()

nifti_bias = preproc.bias_field_correction(str(nifti_file))
nifti_norm = preproc.robust_min_max_scaling(
    str(nifti_file), lower_percentile=1, upper_percentile=99
)

# 3. Débruiter
denoiser_rician = NLMRicianDenoiser()
denoiser_anisotropic = AnisotropicDenoiser()

irm_denoised = denoiser_rician.filter(irm)
irm_denoised = denoiser_anisotropic.filter(irm_denoised)

# 4. Segmentation Otsu via segmed3d
seg = ThresholdSegmentation(irm_denoised.data, irm.affine, irm.header)
mask_otsu = seg(method="otsu")

mask_otsu_propre = Postprocessor.clean_mask(mask_otsu, min_size=50, fill_holes=True)

if mask_otsu_propre.sum() > 0:
    scores = Metrics.all_metrics(mask_otsu_propre, mask)
    print("--- Métriques Otsu vs Reference ---")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

# 5. Morphologie mathématique
nettoyeur = MorphologyCleaning(mask)
mask_propre = nettoyeur.garder_plus_grande_composante()
nettoyeur.set_image(mask_propre)
mask_propre = nettoyeur.supprimer_petits_trous(taille_min=128)

stats_mask = MorphologyStats(mask, mask_propre)
stats_mask.afficher_resume()

morpho = MathMorphology(irm_denoised.data)
gradient = morpho.gradient_morphologique(forme="ball", rayon=1)

enh = MorphologyEnhancement(irm_denoised.data)
tophat = enh.top_hat_blanc(forme="ball", rayon=3)

# 6. Détection des contours
canny = CannyEdgeDetector()
edges_canny = canny.detect(gradient, mask_propre)

sobel = SobelEdgeDetector()
edges_sobel = sobel.detect(gradient, mask_propre)

# 7. Affichage
slice_index = irm.data.shape[2] // 2

slice_original = irm.data[:, :, slice_index]
slice_denoised = irm_denoised.data[:, :, slice_index]
slice_mask = mask[:, :, slice_index]
slice_mask_propre = mask_propre[:, :, slice_index]
slice_gradient = gradient[:, :, slice_index]
slice_tophat = tophat[:, :, slice_index]
slice_canny = edges_canny[:, :, slice_index]
slice_sobel = edges_sobel[:, :, slice_index]
slice_otsu = mask_otsu[:, :, slice_index]
slice_otsu_propre = mask_otsu_propre[:, :, slice_index]
slice_bias = nifti_bias.get_fdata()[:, :, slice_index]
slice_norm = nifti_norm.get_fdata()[:, :, slice_index]

fig, axes = plt.subplots(3, 4, figsize=(20, 14))

axes[0, 0].imshow(slice_original.T, cmap="gray", origin="lower")
axes[0, 0].set_title("Originale")

axes[0, 1].imshow(slice_bias.T, cmap="gray", origin="lower")
axes[0, 1].set_title("Bias field corrected")

axes[0, 2].imshow(slice_norm.T, cmap="gray", origin="lower")
axes[0, 2].set_title("Robust min-max")

axes[0, 3].imshow(slice_denoised.T, cmap="gray", origin="lower")
axes[0, 3].set_title("Débruitée")

axes[1, 0].imshow(slice_mask.T, cmap="gray", origin="lower")
axes[1, 0].set_title("Masque original")

axes[1, 1].imshow(slice_mask_propre.T, cmap="gray", origin="lower")
axes[1, 1].set_title("Masque nettoyé")

axes[1, 2].imshow(slice_gradient.T, cmap="gray", origin="lower")
axes[1, 2].set_title("Gradient morphologique")

axes[1, 3].imshow(slice_tophat.T, cmap="gray", origin="lower")
axes[1, 3].set_title("Top-hat blanc (r=3)")

axes[2, 0].imshow(slice_otsu.T, cmap="gray", origin="lower")
axes[2, 0].set_title("Otsu (segmed3d)")

axes[2, 1].imshow(slice_otsu_propre.T, cmap="gray", origin="lower")
axes[2, 1].set_title("Otsu nettoyé")

axes[2, 2].imshow(slice_canny.T, cmap="gray", origin="lower")
axes[2, 2].set_title("Canny")

axes[2, 3].imshow(slice_sobel.T, cmap="gray", origin="lower")
axes[2, 3].set_title("Sobel")

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.show()
