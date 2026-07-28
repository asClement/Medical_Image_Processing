# Guide d'utilisation complet — Medical Image Processing

## Structure du package `src/`

```
src/
├── preprocessing.py          # Prétraitement d'intensité (scaling, normalisation, CLAHE, N4, GMM)
├── main.py                   # Pipeline d'exemple
├── denoise/                  # Débruitage
├── edges/                    # Détection de contours
├── mathmorpho/               # Morphologie mathématique
├── medio/                    # I/O et structures NIfTI
├── segmed3d/                 # Segmentation (7 algorithmes + utils)
└── utils/                    # Utilitaires (ISPY2Extractor)
```

---

## 1. `medio` — Chargement et sauvegarde NIfTI

```python
from medio.nifti import load_nifti, save_nifti, MedicalImage3D

# Chargement
irm: MedicalImage3D = load_nifti("chemin/vers/image.nii.gz")
print(irm.data.shape)    # (X, Y, Z)
print(irm.affine)        # matrice 4×4
print(irm.spacing)       # (sx, sy, sz)

# Chargement avec masque d'application
irm_masked = load_nifti("image.nii.gz", "mask.nii.gz")

# Sauvegarde
save_nifti(irm.data * 2, "image_scaled.nii.gz", irm.affine, irm.header)
```

---

## 2. `preprocessing.py` — Prétraitement d'intensité

```python
from preprocessing import Preprocessing

pre = Preprocessing()

# Min-max scaling global
img = pre.min_max_global_scaling("input.nii.gz", output_range=(0, 1))

# Robust min-max (percentiles)
img = pre.robust_min_max_scaling("input.nii.gz", lower_percentile=2, upper_percentile=98)

# Z-score normalisation
img = pre.z_score_global_normalization("input.nii.gz")

# Robust z-score (median + IQR)
img = pre.robust_z_score_normalization("input.nii.gz")

# Median-MAD scaling
img = pre.median_mad_scaling("input.nii.gz")

# CLAHE (2D slice-wise)
img = pre.clahe("input.nii.gz", clip_limit=0.02, kernel_size=(32, 32))

# Adaptive Histogram Equalization (SimpleITK 3D)
img = pre.adaptive_histogram_equalization("input.nii.gz", radius=(6, 6, 3))

# Histogram equalization global
img = pre.histogram_equalization("input.nii.gz", nbins=512)

# Gamma correction
img = pre.gamma_correction("input.nii.gz", gamma=0.8)

# N4 Bias field correction
img = pre.bias_field_correction("input.nii.gz", shrink_factor=4)

# Gaussian Mixture Model
img = pre.gaussian_mixture_model("input.nii.gz", n_components=3, return_mode="expected_mean")
img = pre.gaussian_mixture_model("input.nii.gz", return_mode="hard_labels")
img = pre.gaussian_mixture_model("input.nii.gz", return_mode="brightest_posterior")
```

Toutes les méthodes retournent un `nib.Nifti1Image`.

---

## 3. `denoise` — Débruitage

```python
from denoise import AnisotropicDenoiser, NLMRicianDenoiser
from denoise.gaussian import GaussianDenoiser

irm = load_nifti("image.nii.gz")

# Débruitage Rician Non-Local Means (MRI)
denoiser_rician = NLMRicianDenoiser(patch_radius=2, block_radius=6, sigma=0.1)
result = denoiser_rician.filter(irm)

# Débruitage anisotropique (Perona-Malik)
denoiser_aniso = AnisotropicDenoiser(n_iterations=20, conductance=0.5)
result = denoiser_aniso.filter(irm)

# Filtre gaussien
denoiser_gauss = GaussianDenoiser(sigma=1.5)
result = denoiser_gauss.filter(irm)
```

---

## 4. `edges` — Détection de contours

```python
from edges import CannyEdgeDetector, SobelEdgeDetector

irm = load_nifti("image.nii.gz")

# Canny (2D slice-wise pour volumes 3D)
canny = CannyEdgeDetector(sigma=1.0, low_threshold=0.05, high_threshold=0.15)
edges = canny.detect(irm.data)

# Avec masque de restriction
mask = load_nifti("mask.nii.gz").data
edges_masked = canny.detect(irm.data, mask)

# Sobel (3D natif)
sobel = SobelEdgeDetector()
edges = sobel.detect(irm.data)

# Avec masque
edges_masked = sobel.detect(irm.data, mask)
```

---

## 5. `mathmorpho` — Morphologie mathématique

```python
from mathmorpho import (
    MathMorphology, MorphologyCleaning, MorphologyEnhancement,
    MorphologySkeleton, MorphologyShape, MorphologyStats,
    DistanceTransform, WatershedSegmentation,
)

data = load_nifti("image.nii.gz").data

# --- Opérations morphologiques ---
m = MathMorphology(data)
dilated = m.dilatation(forme="ball", rayon=2)
eroded = m.erosion(forme="cross", rayon=1)
opened = m.ouverture(forme="ball", rayon=1)
closed = m.fermeture(forme="ball", rayon=2)
gradient = m.gradient_morphologique(forme="ball", rayon=1)
recon = m.reconstruction(marker, mask)

# --- Nettoyage de masques binaires ---
c = MorphologyCleaning(binary_mask)
cleaned = c.supprimer_petits_objets(taille_min=50)
cleaned = c.supprimer_petits_trous(taille_min=128)
largest = c.garder_plus_grande_composante()
labels = c.etiqueter_composantes()

# --- Rehaussement par top-hat ---
e = MorphologyEnhancement(data)
white = e.top_hat_blanc(forme="ball", rayon=3)    # petites structures claires
black = e.top_hat_noir(forme="ball", rayon=3)      # petites structures sombres

# --- Squelettisation ---
s = MorphologySkeleton(binary_mask)
skel = s.squelettiser()
medial = s.axe_median()

# --- Distance transform ---
d = DistanceTransform(binary_mask)
dist = d.transformee_distance(sampling=(1.0, 1.0, 1.0))

# --- Watershed ---
ws = WatershedSegmentation(data, dist)
labels = ws.segmenter()

# --- Statistiques comparatives ---
s = MorphologyStats(mask_original, mask_clean)
s.afficher_resume()
s.rapport_complet()

# --- Analyse de forme ---
sh = MorphologyShape(binary_mask)
hull = sh.enveloppe_convexe()
convexity = sh.indice_convexite()
```

---

## 6. `segmed3d` — Segmentation (7 algorithmes)

### API commune (scikit-learn style)

```python
from segmed3d import (
    ThresholdSegmentation, WatershedSegmentation, ActiveContourSegmentation,
    LevelSetSegmentation, RegionGrowingSegmentation, ClusteringSegmentation,
    AtlasSegmentation,
)

vol, affine, hdr = load_nifti("image.nii.gz")

# Tous les segmenteurs suivent ce patron :
seg = XxxSegmentation(vol, affine, hdr)
seg.fit(**params)
mask = seg.get_mask()
seg.save("mask.nii.gz")

# Ou en une ligne :
mask = XxxSegmentation(vol, affine, hdr)(**params)
```

### 6.1 ThresholdSegmentation

```python
from segmed3d import ThresholdSegmentation

seg = ThresholdSegmentation(vol, affine, hdr)

# Otsu global
mask = seg(method="otsu")

# Multi-Otsu (2 classes)
mask = seg(method="multi_otsu", n_classes=2)

# Slice-wise Otsu (robuste aux variations inter-coupes)
mask = seg(method="slice_otsu")

# Seuillage manuel
mask = seg(method="manual", low=50, high=200)

print(f"Seuil estimé : {seg.get_threshold():.2f}")
```

### 6.2 WatershedSegmentation

```python
from segmed3d import WatershedSegmentation

seg = WatershedSegmentation(vol, affine, hdr)

# Mode gradient
mask = seg(mode="gradient", sigma=1.0, compactness=0.1)

# Mode distance (automatique depuis un masque binaire)
mask = seg(mode="distance", min_distance=5)

# Mode image directe
mask = seg(mode="image", sigma=1.5)

# Accès aux labels
labels = seg.get_labels()  # avant binarisation
```

### 6.3 ActiveContourSegmentation

```python
from segmed3d import ActiveContourSegmentation

seg = ActiveContourSegmentation(vol, affine, hdr)

# Snakes paramétriques 2D (slice-wise)
mask = seg(method="snakes", alpha=0.01, beta=0.1, n_iterations=100)

# GAC morphologique 3D
mask = seg(method="morphological_gac", sigma=1.0, propagation=10)
```

### 6.4 LevelSetSegmentation

```python
from segmed3d import LevelSetSegmentation

seg = LevelSetSegmentation(vol, affine, hdr)

# Chan-Vese 2D (slice-wise) — rapide, bon pour des formes simples
mask = seg(method="chan_vese_2d", n_iterations=100)

# Chan-Vese 3D morphologique (ACWE) — plus robuste
mask = seg(method="morphological_acwe", n_iterations=50, smoothing=1)

# GAC 3D morphologique
mask = seg(method="morphological_gac", sigma=1.0, propagation=20)
```

### 6.5 RegionGrowingSegmentation

```python
from segmed3d import RegionGrowingSegmentation

seg = RegionGrowingSegmentation(vol, affine, hdr)

# Mode range (tolérance absolue autour du seed)
mask = seg(mode="range", seed=(50, 80, 30), tolerance=20, connectivity=26)

# Mode gradient (tolérance relative aux variations locales)
mask = seg(mode="gradient", seed=(50, 80, 30), tolerance=0.5, connectivity=6)

print(f"Seed utilisé : {seg.get_seed_point()}")
print(f"Valeur au seed : {seg.get_seed_value():.2f}")
```

### 6.6 ClusteringSegmentation

```python
from segmed3d import ClusteringSegmentation

seg = ClusteringSegmentation(vol, affine, hdr)

# K-Means (n_clusters=2 par défaut → fond + objet)
mask = seg(method="kmeans", n_clusters=2, n_init=10)

# Fuzzy C-Means (nécessite scikit-fuzzy)
mask = seg(method="fcm", n_clusters=3, fuzziness=2.0)

# Carte de membre d'un cluster spécifique
membership = seg.get_membership_volume()
cluster_vol = seg.get_cluster_volume(cluster_id=0)
```

### 6.7 AtlasSegmentation (multi-atlas)

```python
from segmed3d import AtlasSegmentation

seg = AtlasSegmentation(target_vol, target_affine, target_header)

# Avec registrations SimpleITK
mask = seg(
    atlas_paths=["atlas1.nii.gz", "atlas2.nii.gz"],
    atlas_mask_paths=["mask1.nii.gz", "mask2.nii.gz"],
    fusion="majority_voting",   # majority_voting | weighted_voting | STAPLE | JLF
    n_jobs=-1,
)

# Sans registration (atlases déjà alignés)
mask = seg(atlas_volumes=[vol1, vol2], atlas_masks=[mask1, mask2],
           fusion="weighted_voting")
```

---

## 7. `segmed3d.utils` — Utilitaires

### ImageIO

```python
from segmed3d import ImageIO

# Chargement
vol, affine, hdr = ImageIO.load_nifti("image.nii.gz")
mask, _, _ = ImageIO.load_mask("mask.nii.gz")     # binarisé > 0 → uint8

# Sauvegarde
ImageIO.save_nifti(vol, "out.nii.gz", affine=affine, header=hdr)
ImageIO.save_mask(mask, "mask_out.nii.gz", affine=affine)

# Espacement voxel
spacing = ImageIO.get_voxel_spacing(affine)
```

### Preprocessor (segmed3d)

```python
from segmed3d import Preprocessor

vol = load_nifti("image.nii.gz").data

# Normalisation [0, 1] ou z-score
norm = Preprocessor.normalize(vol, mode="minmax")
norm = Preprocessor.normalize(vol, mode="zscore")

# Clipping par percentiles
clipped = Preprocessor.clip_intensity(vol, p_low=1, p_high=99)

# Rescaling linéaire
rescaled = Preprocessor.rescale(vol, out_min=0, out_max=255)

# Débruitage
gauss = Preprocessor.gaussian_smooth(vol, sigma=1.0)
median = Preprocessor.median_filter(vol, size=3)
bilateral = Preprocessor.denoise_bilateral(vol, sigma_spatial=1.0)

# N4 Bias field correction
corrected = Preprocessor.bias_field_correction(vol, shrink_factor=2)
```

### Postprocessor

```python
from segmed3d import Postprocessor

mask = load_nifti("mask.nii.gz").data

# Pipeline complet
clean = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True, keep_largest=False)

# Opérations individuelles
filled = Postprocessor.fill_holes(mask, connectivity=26)
cleaned = Postprocessor.remove_small_objects(mask, min_size=50)
largest = Postprocessor.largest_cc(mask)
top_n = Postprocessor.extract_largest_n(mask, n=3)

# Morphologie
opened = Postprocessor.morph_open(mask, radius=2)
closed = Postprocessor.morph_close(mask, radius=2)
eroded = Postprocessor.morph_erode(mask, radius=1)
dilated = Postprocessor.morph_dilate(mask, radius=1)
```

### Metrics

```python
from segmed3d import Metrics

pred = load_nifti("pred.nii.gz").data
gt = load_nifti("gt.nii.gz").data

# Métrique individuelle
dice = Metrics.dice(pred, gt)
iou = Metrics.iou(pred, gt)
sens = Metrics.sensitivity(pred, gt)
spec = Metrics.specificity(pred, gt)
prec = Metrics.precision(pred, gt)

# Hausdorff 95 (avec espacement voxel)
hd = Metrics.hausdorff95(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))

# Volume similarity
vs = Metrics.volume_similarity(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))

# Toutes les métriques en un appel
all_scores = Metrics.all_metrics(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))
# -> {'dice': ..., 'iou': ..., 'sensitivity': ..., 'specificity': ...,
#     'precision': ..., 'hausdorff95': ..., 'volume_similarity': ...}
```

### Visualizer

```python
from segmed3d import Visualizer

vol = load_nifti("image.nii.gz").data
mask = load_nifti("mask.nii.gz").data

# Coupes orthogonales 3×3
Visualizer.plot_3d_slices(vol, mask=mask, n_cols=3, cmap="gray")

# Overlay (masque en rouge semi-transparent sur l'image)
Visualizer.plot_overlay(vol, mask, slice_idx=50, axis=2, alpha=0.4)

# Surface 3D
Visualizer.plot_3d_surface(mask, voxel_spacing=(1.0, 1.0, 1.0))

# Histogramme
Visualizer.plot_histogram(vol, mask=mask, bins=256)
```

---

## 8. Pipeline complet (exemple)

```python
from pathlib import Path
from medio.nifti import load_nifti
from preprocessing import Preprocessing
from denoise import AnisotropicDenoiser, NLMRicianDenoiser
from segmed3d import (
    ThresholdSegmentation, Postprocessor, Metrics,
    ImageIO, Visualizer,
)

# 1. Chargement
vol, affine, hdr = ImageIO.load_nifti("irm.nii.gz")
mask_gt, _, _ = ImageIO.load_mask("reference.nii.gz")

# 2. Prétraitement intensité
pre = Preprocessing()
corrected = pre.bias_field_correction("irm.nii.gz")

# 3. Segmentation Otsu
seg = ThresholdSegmentation(vol, affine, hdr)
pred = seg(method="otsu")

# 4. Post-traitement
pred_clean = Postprocessor.clean_mask(pred, min_size=50, fill_holes=True)

# 5. Évaluation
scores = Metrics.all_metrics(pred_clean, mask_gt,
                              voxel_spacing=ImageIO.get_voxel_spacing(affine))
for k, v in scores.items():
    print(f"{k}: {v:.4f}")

# 6. Visualisation
Visualizer.plot_overlay(vol, pred_clean, slice_idx=vol.shape[2]//2, axis=2)
Visualizer.plot_3d_surface(pred_clean)
```

---

## 9. `utils.ispy2_extractor` — Extraction ISPY2 (TCIA)

```python
from utils.ispy2_extractor import ISPY2Extractor

extractor = ISPY2Extractor(
    project_root=Path("."),
    sample_size=5,
    random_state=42,
)

extractor.info()                    # Aperçu du dataset
extractor.pipeline()                # Pipeline complet

# Étapes individuelles
extractor.download_dicoms()
extractor.convert_series_to_nifti()
extractor.inspect_nifti()
```
