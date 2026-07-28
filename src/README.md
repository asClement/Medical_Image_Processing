# Documentation — Medical Image Processing

> Boîte à outils modulaire pour le traitement et la segmentation d'images médicales 3D (NIfTI).
> Conçue pour les IRM DCE, compatible tout volume 3D.

---

## Table des matières

1. [Vue d'ensemble](#1-vue-densemble)
2. [Tutoriel : pipeline complet](#2-tutoriel--pipeline-complet)
3. [Modules](#3-modules)
   - [medio](#31-medio--io-nifti) — I/O NIfTI
   - [preprocessing](#32-preprocessing--prétraitement-dintensité) — Normalisation, correction, CLAHE
   - [denoise](#33-denoise--débruitage) — Débruitage Rician, anisotropique, gaussien
   - [edges](#34-edges--détection-de-contours) — Canny, Sobel
   - [mathmorpho](#35-mathmorpho--morphologie-mathématique) — Morphologie, nettoyage, squelette, watershed
   - [segmed3d](#36-segmed3d--segmentation) — 7 algorithmes de segmentation
   - [segmed3d.utils](#37-segmed3dutils--utilitaires) — I/O, preprocessing, postprocessing, métriques
4. [Référence rapide](#4-référence-rapide)
5. [Exemples combinés](#5-exemples-combinés)

---

## 1. Vue d'ensemble

Le projet est organisé en **6 packages** sous `src/`, chacun indépendant et importable séparément :

| Package | Rôle |
|---|---|
| `medio` | Chargement/sauvegarde NIfTI, structure `MedicalImage3D` |
| `preprocessing` | Prétraitement d'intensité (scaling, histogramme, N4, GMM) |
| `denoise` | Débruitage 3D (Rician NLM, anisotropique, gaussien) |
| `edges` | Détection de contours (Canny, Sobel) |
| `mathmorpho` | Morphologie mathématique 3D complète |
| `segmed3d` | Segmentation (7 algorithmes) + utilitaires |

Tous les chemins NIfTI acceptent `.nii` ou `.nii.gz`.

---

## 2. Tutoriel : pipeline complet

Ce tutoriel vous guide de l'IRM brute à l'évaluation de la segmentation.

### Prérequis

```bash
pip install numpy scipy scikit-image scikit-learn nibabel SimpleITK matplotlib tqdm
pip install scikit-fuzzy  # optionnel (Fuzzy C-Means)
```

### Étapes

```python
from pathlib import Path
from medio.nifti import load_nifti
from preprocessing import Preprocessing
from denoise import AnisotropicDenoiser, NLMRicianDenoiser
from edges import CannyEdgeDetector
from mathmorpho import MorphologyCleaning, MorphologyStats
from segmed3d import (
    ThresholdSegmentation, Postprocessor, Metrics, ImageIO, Visualizer,
)
```

#### 2.1 Chargement

```python
vol, affine, hdr = ImageIO.load_nifti("irm.nii.gz")
mask_ref, _, _ = ImageIO.load_mask("reference.nii.gz")
print(f"Volume : {vol.shape}, espacement : {ImageIO.get_voxel_spacing(affine)}")
```

#### 2.2 Prétraitement

```python
pre = Preprocessing()
nifti_bias = pre.bias_field_correction("irm.nii.gz", shrink_factor=2)
nifti_norm = pre.robust_min_max_scaling("irm.nii.gz", lower_percentile=1, upper_percentile=99)
```

#### 2.3 Débruitage

```python
irm = load_nifti("irm.nii.gz")
den_rician = NLMRicianDenoiser(patch_radius=2, block_radius=6)
den_aniso = AnisotropicDenoiser(n_iterations=20, conductance=0.5)
irm_den = den_rician.filter(irm)
irm_den = den_aniso.filter(irm_den)
```

#### 2.4 Segmentation (Otsu)

```python
seg = ThresholdSegmentation(irm_den.data, irm.affine, irm.header)
mask_otsu = seg(method="otsu")
print(f"Seuil Otsu : {seg.get_threshold():.2f}")
```

#### 2.5 Post-traitement du masque

```python
mask_clean = Postprocessor.clean_mask(mask_otsu, min_size=50, fill_holes=True)
```

#### 2.6 Nettoyage morphologique du masque de référence

```python
c = MorphologyCleaning(mask_ref)
mask_ref_propre = c.garder_plus_grande_composante()
mask_ref_propre = c.supprimer_petits_trous(taille_min=128)
stats = MorphologyStats(mask_ref, mask_ref_propre)
stats.afficher_resume()
```

#### 2.7 Évaluation

```python
spacing = ImageIO.get_voxel_spacing(affine)
scores = Metrics.all_metrics(mask_clean, mask_ref_propre, voxel_spacing=spacing)
for k, v in scores.items():
    print(f"  {k}: {v:.4f}")
```

#### 2.8 Visualisation

```python
Visualizer.plot_3d_slices(vol, mask=mask_clean, n_cols=4)
Visualizer.plot_overlay(vol, mask_clean, slice_idx=vol.shape[2] // 2, axis=2)
```

---

## 3. Modules

### 3.1 `medio` — I/O NIfTI

```python
from medio.nifti import load_nifti, save_nifti, MedicalImage3D
```

**`load_nifti(path, mask_path=None)`** → `MedicalImage3D`

- Retourne un objet avec `.data` (ndarray), `.affine` (4×4), `.header`, `.spacing`
- Si `mask_path` est fourni, les voxels hors masque sont mis à 0

**`save_nifti(data, path, affine=None, header=None)`**

Sauvegarde un tableau 3D au format NIfTI.

```python
irm = load_nifti("img.nii.gz")
print(irm.data.shape, irm.spacing)  # (512, 512, 120) (0.5, 0.5, 1.2)
```

---

### 3.2 `preprocessing` — Prétraitement d'intensité

```python
from preprocessing import Preprocessing
```

Classe instanciable. Chaque méthode prend un **chemin NIfTI** et retourne un `nib.Nifti1Image`.

| Méthode | Effet |
|---|---|
| `min_max_global_scaling` | Mise à l'échelle linéaire `[min, max] → [a, b]` |
| `robust_min_max_scaling` | Percentiles comme bornes (robuste aux outliers) |
| `z_score_global_normalization` | Standardisation `(x - μ) / σ` |
| `robust_z_score_normalization` | `(x - median) / (IQR / 1.349)` |
| `median_mad_scaling` | `(x - median) / MAD (×1.4826)` |
| `clahe` | CLAHE 2D slice-wise (via scikit-image) |
| `adaptive_histogram_equalization` | AHE 3D (via SimpleITK) |
| `histogram_equalization` | Égalisation globale par CDF |
| `gamma_correction` | Correction gamma avec normalisation locale |
| `bias_field_correction` | N4 bias field correction (SimpleITK) |
| `gaussian_mixture_model` | GMM 1D sur intensités (3 modes de retour) |

Toutes les méthodes supportent les paramètres :
- `foreground_threshold` (défaut `0.0`) — seuil du masque de fond
- `preserve_background` (défaut `True`) — préserver les voxels hors premier plan

```python
pre = Preprocessing()
img = pre.bias_field_correction("irm.nii.gz", shrink_factor=4)
img = pre.clahe("irm.nii.gz", clip_limit=0.02, kernel_size=(32, 32))
img = pre.gaussian_mixture_model("irm.nii.gz", n_components=3,
                                  return_mode="brightest_posterior")
```

---

### 3.3 `denoise` — Débruitage

```python
from denoise import AnisotropicDenoiser, NLMRicianDenoiser
from denoise.gaussian import GaussianDenoiser
```

Tous les débruiteurs héritent de `BaseMedicalDenoiser` et implémentent `filter(medical_image)`.

| Classe | Algorithme | Usage |
|---|---|---|
| `NLMRicianDenoiser` | Non-Local Means adapté au bruit Rician (IRM) | `patch_radius`, `block_radius`, `sigma` |
| `AnisotropicDenoiser` | Diffusion anisotropique (Perona-Malik) | `n_iterations`, `conductance`, `time_step` |
| `GaussianDenoiser` | Filtre gaussien (scipy) | `sigma` |

```python
den = AnisotropicDenoiser(n_iterations=30, conductance=0.3)
irm_filtre = den.filter(irm)
```

---

### 3.4 `edges` — Détection de contours

```python
from edges import CannyEdgeDetector, SobelEdgeDetector
```

**`CannyEdgeDetector(sigma=1.0, low_threshold=0.05, high_threshold=0.15)`**

- Applique Canny slice par slice sur les volumes 3D
- `detect(data, mask=None)` → ndarray binaire

**`SobelEdgeDetector()`**

- Gradient Sobel 3D natif (scipy.ndimage.sobel)
- `detect(data, mask=None)` → ndarray gradient magnitude

```python
edges = CannyEdgeDetector(sigma=0.8).detect(irm.data, mask)
edges = SobelEdgeDetector().detect(irm.data)
```

---

### 3.5 `mathmorpho` — Morphologie mathématique

```python
from mathmorpho import (
    MathMorphology, MorphologyCleaning, MorphologyEnhancement,
    MorphologySkeleton, MorphologyShape, MorphologyStats,
    DistanceTransform, WatershedSegmentation,
)
```

#### Opérations morphologiques (`MathMorphology`)

| Méthode | Description |
|---|---|
| `dilatation(forme, rayon)` | Dilatation 3D |
| `erosion(forme, rayon)` | Érosion 3D |
| `ouverture(forme, rayon)` | Ouverture (érosion + dilatation) |
| `fermeture(forme, rayon)` | Fermeture (dilatation + érosion) |
| `gradient_morphologique(forme, rayon)` | Gradient = dilatation - érosion |
| `reconstruction(marker, mask)` | Reconstruction géodésique |
| `erosion_geodesique(marker, mask, n_iter)` | Érosion géodésique itérative |

`forme` ∈ `{"ball", "cross", "cube"}`

#### Nettoyage (`MorphologyCleaning`)

| Méthode | Description |
|---|---|
| `supprimer_petits_objets(taille_min)` | Supprime les composantes < taille_min |
| `supprimer_petits_trous(taille_min)` | Comble les trous < taille_min |
| `garder_plus_grande_composante()` | Garde la plus grande CC |
| `etiqueter_composantes(connectivity)` | Labelisation des CC |

#### Rehaussement (`MorphologyEnhancement`)

| Méthode | Description |
|---|---|
| `top_hat_blanc(forme, rayon)` | Structures claires sur fond sombre |
| `top_hat_noir(forme, rayon)` | Structures sombres sur fond clair |

#### Squelette (`MorphologySkeleton`)

| Méthode | Description |
|---|---|
| `squelettiser()` | Squelette topologique 2D/3D (scipy) |
| `axe_median()` | Axe médian 2D (skimage) |

#### Distance (`DistanceTransform`)

| Méthode | Description |
|---|---|
| `transformee_distance(sampling)` | Distance euclidienne avec espacement voxel |

#### Watershed (`WatershedSegmentation`)

```python
ws = WatershedSegmentation(image, distance)
labels = ws.segmenter(marqueurs=None)  # marqueurs automatiques si None
```

#### Analyse de forme (`MorphologyShape`)

| Méthode | Description |
|---|---|
| `enveloppe_convexe()` | Enveloppe convexe 3D (scipy) |
| `indice_convexite()` | Ratio volume / volume enveloppe convexe |

#### Statistiques (`MorphologyStats`)

```python
s = MorphologyStats(mask_avant, mask_apres)
s.afficher_resume()                        # volume, Δ, précision/rappel
s.histogramme_intensites(image)            # histogrammes avant/après
s.rapport_complet(image, slice_idx)        # tout en un
```

---

### 3.6 `segmed3d` — Segmentation

API scikit-learn commune :

```python
seg = XxxSegmentation(volume, affine, header)
mask = seg(**params)           # fit() + get_mask()
seg.save("mask.nii.gz")
```

| Classe | Méthodes | Description |
|---|---|---|
| **ThresholdSegmentation** | `otsu`, `multi_otsu`, `slice_otsu`, `manual` | Seuillage automatique ou manuel |
| **WatershedSegmentation** | `gradient`, `distance`, `image` | Watershed par marqueurs |
| **ActiveContourSegmentation** | `snakes`, `morphological_gac` | Contours actifs paramétriques ou GAC 3D |
| **LevelSetSegmentation** | `chan_vese_2d`, `morphological_acwe`, `morphological_gac` | Level set Chan-Vese ou GAC |
| **RegionGrowingSegmentation** | `range`, `gradient` | Croissance de région avec seed |
| **ClusteringSegmentation** | `kmeans`, `fcm` | K-Means ou Fuzzy C-Means |
| **AtlasSegmentation** | multi-atlas avec fusion | Majority voting, weighted, STAPLE, JLF |

```python
# Exemple : watershed
from segmed3d import WatershedSegmentation
seg = WatershedSegmentation(vol, affine, hdr)
mask = seg(mode="gradient", sigma=1.0)

# Exemple : level set
from segmed3d import LevelSetSegmentation
seg = LevelSetSegmentation(vol, affine, hdr)
mask = seg(method="morphological_acwe", n_iterations=50)
```

---

### 3.7 `segmed3d.utils` — Utilitaires

#### ImageIO

| Méthode statique | Description |
|---|---|
| `load_nifti(path)` | → `(volume, affine, header)` |
| `load_mask(path)` | → `(mask_bin, affine, header)` |
| `save_nifti(vol, path, affine, header)` | Sauvegarde volume |
| `save_mask(mask, path, affine, header)` | Sauvegarde masque binarisé |
| `get_voxel_spacing(affine)` | → `(sx, sy, sz)` |

#### Preprocessor

| Méthode statique | Description |
|---|---|
| `normalize(vol, mode)` | Min-max ou z-score sur tableau numpy |
| `clip_intensity(vol, p_low, p_high)` | Clipping percentile |
| `rescale(vol, out_min, out_max)` | Rescaling linéaire |
| `gaussian_smooth(vol, sigma)` | Filtre gaussien scipy |
| `median_filter(vol, size)` | Filtre médian 3D |
| `denoise_bilateral(vol, sigma_spatial)` | Bilateral 2D slice-wise |
| `bias_field_correction(vol, ...)` | N4 correction sur tableau numpy |

#### Postprocessor

| Méthode statique | Description |
|---|---|
| `fill_holes(mask, connectivity)` | Comblement des trous 3D |
| `remove_small_objects(mask, min_size)` | Suppression petites CC |
| `largest_cc(mask)` | Plus grande composante |
| `extract_largest_n(mask, n)` | N plus grandes CC |
| `clean_mask(mask, ...)` | Pipeline tout-en-un (fill + remove + largest) |
| `morph_open / morph_close / morph_erode / morph_dilate` | Morphologie sur masque |

#### Metrics

| Méthode statique | Description |
|---|---|
| `dice(pred, gt)` | Dice coefficient |
| `iou(pred, gt)` | Intersection over Union |
| `sensitivity / specificity / precision` | Métriques de classification |
| `hausdorff95(pred, gt, voxel_spacing)` | Distance de Hausdorff 95e percentile |
| `volume_similarity(pred, gt, voxel_spacing)` | Similarité volumique |
| `all_metrics(pred, gt, voxel_spacing)` | Toutes les métriques en un dict |

#### Visualizer

| Méthode statique | Description |
|---|---|
| `plot_3d_slices(vol, mask, n_cols)` | Grille de coupes orthogonales |
| `plot_overlay(vol, mask, slice_idx, axis)` | Overlay masque sur image |
| `plot_3d_surface(mask, voxel_spacing)` | Rendu de surface 3D |
| `plot_histogram(vol, mask, bins)` | Histogramme d'intensités |

```python
from segmed3d import Preprocessor, Postprocessor, Metrics, Visualizer

vol = load_nifti("img.nii.gz").data
vol = Preprocessor.normalize(vol, mode="minmax")
mask = Postprocessor.clean_mask(mask_raw, min_size=50, fill_holes=True)
dice = Metrics.dice(mask, mask_ref)
Visualizer.plot_overlay(vol, mask, slice_idx=50)
```

---

## 4. Référence rapide

### Arbre d'import

```
src/
├── preprocessing.py
│   └── Preprocessing          # 11 méthodes de normalisation/correction
├── denoise/
│   ├── AnisotropicDenoiser    # Diffusion anisotropique
│   ├── NLMRicianDenoiser      # NLM Rician (IRM)
│   └── GaussianDenoiser       # Filtre gaussien
├── edges/
│   ├── CannyEdgeDetector      # Canny 2D slice-wise
│   └── SobelEdgeDetector      # Sobel 3D
├── mathmorpho/
│   ├── MathMorphology         # dilatation, érosion, gradient, reconstruction
│   ├── MorphologyCleaning     # supprimer petits objets/trous, plus grande CC
│   ├── MorphologyEnhancement  # top-hat blanc/noir
│   ├── MorphologySkeleton     # squelette, axe médian
│   ├── MorphologyShape        # enveloppe convexe, convexité
│   ├── MorphologyStats        # statistiques comparatives
│   ├── DistanceTransform      # distance euclidienne
│   └── WatershedSegmentation  # watershed par marqueurs
├── medio/
│   └── load_nifti / save_nifti / MedicalImage3D
├── segmed3d/
│   ├── ThresholdSegmentation  # otsu, multi_otsu, slice_otsu, manual
│   ├── WatershedSegmentation  # gradient, distance, image
│   ├── ActiveContourSegmentation # snakes, morphological_gac
│   ├── LevelSetSegmentation   # chan_vese_2d, morphological_acwe, morphological_gac
│   ├── RegionGrowingSegmentation # range, gradient
│   ├── ClusteringSegmentation # kmeans, fcm
│   ├── AtlasSegmentation      # multi-atlas avec fusion
│   └── utils/
│       ├── ImageIO            # load/save NIfTI
│       ├── Preprocessor       # normalize, clip, smooth, N4
│       ├── Postprocessor      # fill, remove, clean, morph
│       ├── Metrics            # dice, iou, hd95, ...
│       └── Visualizer         # slices, overlay, surface, hist
```

---

## 5. Exemples combinés

### Pipeline complet — segmentation et évaluation

```python
from segmed3d import ImageIO, ThresholdSegmentation, Postprocessor, Metrics

vol, affine, hdr = ImageIO.load_nifti("irm.nii.gz")
gt, _, _ = ImageIO.load_mask("reference.nii.gz")

mask = ThresholdSegmentation(vol, affine, hdr)(method="otsu")
mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)

scores = Metrics.all_metrics(mask, gt, ImageIO.get_voxel_spacing(affine))
```

### Rehaussement + watershed

```python
from mathmorpho import MathMorphology, MorphologyEnhancement, WatershedSegmentation
from mathmorpho import DistanceTransform

data = load_nifti("irm.nii.gz").data

# Rehaussement
enh = MorphologyEnhancement(data)
tophat = enh.top_hat_blanc(forme="ball", rayon=5)

# Gradient morphologique
m = MathMorphology(tophat)
gradient = m.gradient_morphologique(forme="ball", rayon=1)

# Watershed
dist = DistanceTransform(gradient > 0.2)
ws = WatershedSegmentation(gradient, dist.transformee_distance())
segmentation = ws.segmenter()
```

### Débruitage + contours actifs

```python
from denoise import NLMRicianDenoiser
from segmed3d import ActiveContourSegmentation

irm = load_nifti("irm.nii.gz")
den = NLMRicianDenoiser(patch_radius=2, block_radius=6).filter(irm)

seg = ActiveContourSegmentation(den.data, irm.affine, irm.header)
mask = seg(method="snakes", alpha=0.05, beta=0.1, n_iterations=150)
```

### Nettoyage et statistiques morphologiques

```python
from mathmorpho import MorphologyCleaning, MorphologyStats, MorphologyShape

c = MorphologyCleaning(mask)
propre = c.garder_plus_grande_composante()
propre = c.supprimer_petits_trous(128)

MorphologyStats(mask, propre).rapport_complet(image_data)

shape = MorphologyShape(propre)
print(f"Convexité : {shape.indice_convexite():.3f}")
```
