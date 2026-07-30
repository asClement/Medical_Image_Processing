import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.measure import marching_cubes

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
RESULTS_FOLDER = Path("results")
SLICE_X = 108
SLICE_Y = 87
SLICE_Z = 27
nifti_file = (
    Path(NIFTI_FOLDER)
    / f"{PATIENT_ID}/{SESSION_ID}/perf/{PATIENT_ID}_{SESSION_ID}_{SERIES_ID}.nii.gz"
)
mask_file = (
    Path(NIFTI_FOLDER) / f"{PATIENT_ID}/{SESSION_ID}/seg/{PATIENT_ID}_{SESSION_ID}_mask.nii.gz"
)

RESULTS_DIR = RESULTS_FOLDER / PATIENT_ID / SESSION_ID
RESULTS_DIRS = {
    "preprocessing": RESULTS_DIR / "preprocessing",
    "denoising": RESULTS_DIR / "denoising",
    "segmentation": RESULTS_DIR / "segmentation",
    "morphology": RESULTS_DIR / "morphology",
    "edges": RESULTS_DIR / "edges",
    "figures": RESULTS_DIR / "figures",
    "metrics": RESULTS_DIR / "metrics",
    "code": RESULTS_DIR / "code",
}
for directory in RESULTS_DIRS.values():
    directory.mkdir(parents=True, exist_ok=True)
FIGURES_2D_DIR = RESULTS_DIRS["figures"] / "2d"
FIGURES_2D_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_MPR_DIR = RESULTS_DIRS["figures"] / "multiplanar"
FIGURES_MPR_DIR.mkdir(parents=True, exist_ok=True)


def save_volume(data: np.ndarray, output_path: Path, reference: nib.Nifti1Image) -> None:
    """Save a volume while preserving the spatial metadata of a reference image."""
    header = reference.header.copy()
    header.set_data_dtype(np.asarray(data).dtype)
    image = nib.Nifti1Image(np.asarray(data), reference.affine, header=header)
    nib.save(image, str(output_path))


def save_2d_slice(
    data: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str = "gray",
    binary: bool = False,
    slice_index: int = SLICE_Z,
) -> None:
    """Save the central axial slice of a 3D result as a PNG image."""
    volume = np.asarray(data)
    slice_index = min(slice_index, volume.shape[2] - 1)
    slice_data = volume[:, :, slice_index]

    fig, ax = plt.subplots(figsize=(7, 7))
    if binary:
        ax.imshow(slice_data.T, cmap=cmap, origin="lower", vmin=0, vmax=1)
    else:
        finite_values = slice_data[np.isfinite(slice_data)]
        if finite_values.size:
            vmin, vmax = np.percentile(finite_values, [1, 99])
            if vmin == vmax:
                vmin, vmax = float(finite_values.min()), float(finite_values.max())
            ax.imshow(slice_data.T, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(slice_data.T, cmap=cmap, origin="lower")
    ax.set_title(f"{title} — coupe axiale z={slice_index}")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _display_limits(data: np.ndarray) -> tuple[float, float]:
    """Return robust display limits for a scalar volume."""
    values = np.asarray(data)[np.isfinite(data)]
    if values.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(values, [1, 99])
    if vmin == vmax:
        vmin, vmax = float(values.min()), float(values.max())
    return float(vmin), float(vmax) if vmax > vmin else float(vmin + 1.0)


def _plot_plane(
    ax: plt.Axes,
    image: np.ndarray,
    background: np.ndarray | None,
    plane: str,
    index: int,
    cross_x: int,
    cross_y: int,
    title: str,
    cmap: str,
    overlay_cmap: str = "autumn",
) -> None:
    """Plot one orthogonal plane, optionally over a grayscale background."""
    if plane == "axial":
        result_slice = image[:, :, index].T
        background_slice = None if background is None else background[:, :, index].T
        horizontal_cross, vertical_cross = cross_x, cross_y
    elif plane == "coronal":
        result_slice = image[:, index, :].T
        background_slice = None if background is None else background[:, index, :].T
        horizontal_cross, vertical_cross = cross_x, cross_y
    else:  # sagittal
        result_slice = image[index, :, :].T
        background_slice = None if background is None else background[index, :, :].T
        horizontal_cross, vertical_cross = cross_x, cross_y

    if background_slice is not None:
        bg_min, bg_max = _display_limits(background_slice)
        ax.imshow(
            background_slice,
            cmap="gray",
            origin="lower",
            vmin=bg_min,
            vmax=bg_max,
            interpolation="nearest",
        )
        overlay = np.ma.masked_where(result_slice <= 0, result_slice)
        ax.imshow(overlay, cmap=overlay_cmap, origin="lower", alpha=0.55, interpolation="nearest")
    else:
        vmin, vmax = _display_limits(result_slice)
        ax.imshow(
            result_slice,
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

    ax.axvline(horizontal_cross, color="red", linewidth=0.8)
    ax.axhline(vertical_cross, color="red", linewidth=0.8)
    ax.set_title(title, color="black", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def save_multiplanar_view(
    result: np.ndarray,
    output_path: Path,
    title: str,
    background: np.ndarray | None = None,
    binary: bool = False,
    cmap: str = "viridis",
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> None:
    """Save a Slicer-like orthogonal view with a 3D surface rendering."""
    result = np.asarray(result)
    if result.ndim != 3:
        raise ValueError("A multiplanar view requires a 3D volume.")

    display_result = result > 0 if binary else result

    x = min(SLICE_X, result.shape[0] - 1)
    y = min(SLICE_Y, result.shape[1] - 1)
    z = min(SLICE_Z, result.shape[2] - 1)

    fig = plt.figure(figsize=(13, 10), facecolor="white")
    grid = fig.add_gridspec(2, 2, wspace=0.02, hspace=0.12)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0])]

    _plot_plane(
        axes[0], display_result, background, "axial", z, x, y, f"Axial — z={z}", cmap
    )
    _plot_plane(
        axes[1], display_result, background, "coronal", y, x, z, f"Coronal — y={y}", cmap
    )
    _plot_plane(
        axes[2], display_result, background, "sagittal", x, y, z, f"Sagittal — x={x}", cmap
    )

    ax3d = fig.add_subplot(grid[1, 1], projection="3d")
    ax3d.set_facecolor("white")
    if binary:
        surface_volume = display_result.astype(np.float32)
        surface_color = "#ffd900"
    else:
        nonzero = result[np.isfinite(result) & (result != 0)]
        threshold = float(np.percentile(nonzero, 85)) if nonzero.size else 0.0
        surface_volume = (result >= threshold).astype(np.float32)
        surface_color = "#f2d400"

    if surface_volume.any() and not np.all(surface_volume):
        try:
            vertices, faces, _, _ = marching_cubes(
                surface_volume,
                level=0.5,
                spacing=spacing,
                step_size=2,
            )
            ax3d.plot_trisurf(
                vertices[:, 0],
                vertices[:, 1],
                vertices[:, 2],
                triangles=faces,
                color=surface_color,
                alpha=0.78,
                linewidth=0,
            )
        except (ValueError, RuntimeError):
            ax3d.text2D(0.28, 0.5, "Surface 3D indisponible", color="black", transform=ax3d.transAxes)
    else:
        ax3d.text2D(0.25, 0.5, "Surface 3D indisponible", color="black", transform=ax3d.transAxes)

    ax3d.set_title("Rendu 3D", color="black", fontsize=11)
    ax3d.set_xlabel("X", color="black")
    ax3d.set_ylabel("Y", color="black")
    ax3d.set_zlabel("Z", color="black")
    ax3d.tick_params(colors="black", labelsize=7)
    ax3d.view_init(elev=22, azim=-65)

    fig.suptitle(title, color="black", fontsize=15, y=0.98)
    fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def snapshot_code(output_directory: Path) -> None:
    """Copy the source files used by the run into the results directory."""
    source_files = [Path("main.py"), Path("pyproject.toml"), Path("README.md")]
    source_files.extend(Path("src").rglob("*.py"))

    for source_file in source_files:
        if not source_file.is_file():
            continue
        destination = output_directory / source_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, destination)


reference_img = nib.load(str(nifti_file))
mask = load_nifti(mask_file).data
# Conserver l'image complète pendant le prétraitement et le débruitage.
# Le masque est utilisé séparément pour la ROI, les superpositions et l'évaluation.
irm_original = load_nifti(nifti_file)

# 2. Prétraitement d'intensité (preprocessing.py)
preproc = Preprocessing()

nifti_bias = preproc.bias_field_correction(str(nifti_file))
nifti_norm = preproc.robust_min_max_scaling(
    str(nifti_file), lower_percentile=1, upper_percentile=99
)
save_volume(
    nifti_bias.get_fdata(dtype=np.float32),
    RESULTS_DIRS["preprocessing"] / "dce-post-1_bias_corrected.nii.gz",
    reference_img,
)
save_volume(
    nifti_norm.get_fdata(dtype=np.float32),
    RESULTS_DIRS["preprocessing"] / "dce-post-1_robust_minmax.nii.gz",
    reference_img,
)

# 3. Débruiter
denoiser_rician = NLMRicianDenoiser(use_mask=False)
denoiser_anisotropic = AnisotropicDenoiser()

irm_denoised = denoiser_rician.filter(irm_original)
irm_denoised = denoiser_anisotropic.filter(irm_denoised)
save_volume(
    irm_denoised.data.astype(np.float32),
    RESULTS_DIRS["denoising"] / "dce-post-1_nlm_rician_anisotropic.nii.gz",
    reference_img,
)

# 4. Segmentation Otsu via segmed3d
seg = ThresholdSegmentation(irm_denoised.data, irm_original.affine, irm_original.header)
mask_otsu = seg(method="otsu")

mask_otsu_propre = Postprocessor.clean_mask(mask_otsu, min_size=50, fill_holes=True)

save_volume(
    mask_otsu.astype(np.uint8),
    RESULTS_DIRS["segmentation"] / "dce-post-1_otsu.nii.gz",
    reference_img,
)
save_volume(
    mask_otsu_propre.astype(np.uint8),
    RESULTS_DIRS["segmentation"] / "dce-post-1_otsu_cleaned.nii.gz",
    reference_img,
)

if mask_otsu_propre.sum() > 0:
    scores = Metrics.all_metrics(mask_otsu_propre, mask, voxel_spacing=irm_original.spacing)
    print("--- Métriques Otsu vs Reference ---")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")
    (RESULTS_DIRS["metrics"] / "otsu_vs_reference.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )

# 5. Morphologie mathématique
nettoyeur = MorphologyCleaning(mask)
mask_propre = nettoyeur.garder_plus_grande_composante()
nettoyeur.set_image(mask_propre)
mask_propre = nettoyeur.supprimer_petits_trous(taille_min=128)
save_volume(
    mask_propre.astype(np.uint8),
    RESULTS_DIRS["morphology"] / "reference_mask_cleaned.nii.gz",
    reference_img,
)

stats_mask = MorphologyStats(mask, mask_propre)
stats_mask.afficher_resume()

morpho = MathMorphology(irm_denoised.data)
gradient = morpho.gradient_morphologique(forme="ball", rayon=1)

enh = MorphologyEnhancement(irm_denoised.data)
tophat = enh.top_hat_blanc(forme="ball", rayon=3)
save_volume(
    gradient.astype(np.float32),
    RESULTS_DIRS["morphology"] / "morphological_gradient.nii.gz",
    reference_img,
)
save_volume(
    tophat.astype(np.float32),
    RESULTS_DIRS["morphology"] / "white_tophat_radius3.nii.gz",
    reference_img,
)

# 6. Détection des contours
canny = CannyEdgeDetector()
edges_canny = canny.detect(gradient, mask_propre)

sobel = SobelEdgeDetector()
edges_sobel = sobel.detect(gradient, mask_propre)
save_volume(
    edges_canny.astype(np.uint8),
    RESULTS_DIRS["edges"] / "canny.nii.gz",
    reference_img,
)
save_volume(
    edges_sobel.astype(np.uint8),
    RESULTS_DIRS["edges"] / "sobel.nii.gz",
    reference_img,
)

# 7. Affichage
slice_index = min(SLICE_Z, irm_original.data.shape[2] - 1)

slice_original = irm_original.data[:, :, slice_index]
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

images_2d = {
    "01_original.png": (irm_original.data, "Image originale", "gray", False),
    "02_bias_corrected.png": (
        nifti_bias.get_fdata(dtype=np.float32),
        "Correction de biais",
        "gray",
        False,
    ),
    "03_robust_minmax.png": (
        nifti_norm.get_fdata(dtype=np.float32),
        "Normalisation robuste min-max",
        "gray",
        False,
    ),
    "04_denoised.png": (
        irm_denoised.data,
        "Débruitage NLM Rician + anisotrope",
        "gray",
        False,
    ),
    "05_reference_mask.png": (mask, "Masque de référence", "viridis", False),
    "06_cleaned_mask.png": (mask_propre, "Masque nettoyé", "gray", True),
    "07_morphological_gradient.png": (
        gradient,
        "Gradient morphologique",
        "magma",
        False,
    ),
    "08_white_tophat.png": (tophat, "Top-hat blanc", "magma", False),
    "09_otsu.png": (mask_otsu, "Segmentation Otsu", "gray", True),
    "10_otsu_cleaned.png": (mask_otsu_propre, "Segmentation Otsu nettoyée", "gray", True),
    "11_canny.png": (edges_canny, "Contours Canny", "gray", True),
    "12_sobel.png": (edges_sobel, "Contours Sobel", "gray", True),
}
for filename, (data, title, cmap, binary) in images_2d.items():
    save_2d_slice(
        data,
        FIGURES_2D_DIR / filename,
        title,
        cmap=cmap,
        binary=binary,
        slice_index=SLICE_Z,
    )

images_multiplanar = {
    "01_original.png": (irm_original.data, "Image originale", None, False, "gray"),
    "02_bias_corrected.png": (
        nifti_bias.get_fdata(dtype=np.float32),
        "Correction de biais",
        None,
        False,
        "viridis",
    ),
    "03_robust_minmax.png": (
        nifti_norm.get_fdata(dtype=np.float32),
        "Normalisation robuste min-max",
        None,
        False,
        "viridis",
    ),
    "04_denoised.png": (
        irm_denoised.data,
        "Débruitage NLM Rician + anisotrope",
        None,
        False,
        "viridis",
    ),
    "05_reference_mask.png": (mask, "Masque de référence", irm_original.data, True, "gray"),
    "06_cleaned_mask.png": (mask_propre, "Masque nettoyé", irm_original.data, True, "gray"),
    "07_morphological_gradient.png": (
        gradient,
        "Gradient morphologique",
        None,
        False,
        "magma",
    ),
    "08_white_tophat.png": (tophat, "Top-hat blanc", None, False, "magma"),
    "09_otsu.png": (mask_otsu, "Segmentation Otsu", irm_original.data, True, "gray"),
    "10_otsu_cleaned.png": (
        mask_otsu_propre,
        "Segmentation Otsu nettoyée",
        irm_original.data,
        True,
        "gray",
    ),
    "11_canny.png": (edges_canny, "Contours Canny", irm_original.data, True, "gray"),
    "12_sobel.png": (edges_sobel, "Contours Sobel", irm_original.data, True, "gray"),
}
for filename, (data, title, background, binary, cmap) in images_multiplanar.items():
    save_multiplanar_view(
        data,
        FIGURES_MPR_DIR / filename,
        title,
        background=background,
        binary=binary,
        cmap=cmap,
        spacing=irm_original.spacing,
    )

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
fig.savefig(RESULTS_DIRS["figures"] / "pipeline_overview.png", dpi=200, bbox_inches="tight")
plt.close(fig)

metadata = {
    "patient": PATIENT_ID,
    "session": SESSION_ID,
    "source_image": str(nifti_file),
    "source_mask": str(mask_file),
    "shape": list(irm_original.data.shape),
    "spacing_mm": list(irm_original.spacing),
    "mask_applied_before_preprocessing": False,
    "mask_usage": "ROI, overlays, morphology and evaluation only",
    "slice_coordinates": {"x": SLICE_X, "y": SLICE_Y, "z": SLICE_Z},
    "results_directory": str(RESULTS_DIR),
}
(RESULTS_DIR / "run_metadata.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
)
snapshot_code(RESULTS_DIRS["code"])
print(f"Résultats enregistrés dans : {RESULTS_DIR}")
