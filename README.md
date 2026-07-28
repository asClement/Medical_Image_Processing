# Medical Image Processing

Boîte à outils modulaire pour le traitement, la segmentation et l'analyse d'images médicales 3D (NIfTI).  
Conçue pour les IRM DCE, compatible tout volume 3D.

## Structure du projet

```
├── main.py                 # Pipeline principal (prétraitement → débruitage → segmentation → contours)
├── src/
│   ├── main.py             # Point d'entrée : extraction ISPY2 depuis TCIA
│   ├── preprocessing.py    # Prétraitement d'intensité
│   ├── denoise/            # Débruitage Rician NLM, anisotropique, gaussien
│   ├── edges/              # Détection de contours (Canny, Sobel)
│   ├── mathmorpho/         # Morphologie mathématique 3D complète
│   ├── medio/              # I/O NIfTI (MedicalImage3D)
│   ├── segmed3d/           # 7 algorithmes de segmentation + utilitaires
│   │   └── utils/          # I/O, preprocessing, postprocessing, metrics, viz
│   └── utils/              # Extraction ISPY2 (TCIA)
├── notebooks/              # Jupyter notebooks d'analyse
├── tests/                  # Tests unitaires
├── pyproject.toml          # Configuration du projet
└── README.md
```

## Installation

```bash
# Édition développeur (recommandé)
pip install -e .

# Avec support Fuzzy C-Means
pip install -e ".[fuzzy]"

# Dépendances dev (tests, linting)
pip install -e ".[dev]"
```

## Utilisation rapide

```python
from medio.nifti import load_nifti
from preprocessing import Preprocessing
from segmed3d import ThresholdSegmentation, Postprocessor, Metrics

vol, affine, hdr = load_nifti("irm.nii.gz")
gt, _, _ = load_nifti("reference.nii.gz")

mask = ThresholdSegmentation(vol, affine, hdr)(method="otsu")
mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)

print(f"Dice: {Metrics.dice(mask, gt):.4f}")
```

## Points d'entrée

| Commande | Description |
|---|---|
| `python main.py` | Pipeline complet IRM DCE (chargement → débruitage → segmentation Otsu → contours → visualisation) |
| `python -m src.main` | Extraction du dataset ISPY2 depuis TCIA (téléchargement DICOM → conversion NIfTI → organisation BIDS) |

## Packages

| Package | Classes principales | Description |
|---|---|---|
| `medio` | `load_nifti`, `save_nifti`, `MedicalImage3D` | I/O NIfTI avec espacement voxel |
| `preprocessing` | `Preprocessing` (11 méthodes) | Normalisation, CLAHE, N4 bias correction, GMM |
| `denoise` | `AnisotropicDenoiser`, `NLMRicianDenoiser`, `GaussianDenoiser` | Débruitage 3D adapté à l'IRM |
| `edges` | `CannyEdgeDetector`, `SobelEdgeDetector` | Détection de contours 2D/3D |
| `mathmorpho` | `MathMorphology`, `MorphologyCleaning`, `MorphologyEnhancement`, `MorphologySkeleton`, `MorphologyShape`, `MorphologyStats`, `DistanceTransform`, `WatershedSegmentation` | Morphologie mathématique 3D complète |
| `segmed3d` | `ThresholdSegmentation`, `WatershedSegmentation`, `ActiveContourSegmentation`, `LevelSetSegmentation`, `RegionGrowingSegmentation`, `ClusteringSegmentation`, `AtlasSegmentation` | 7 algorithmes de segmentation (API scikit-learn) |
| `segmed3d.utils` | `ImageIO`, `Preprocessor`, `Postprocessor`, `Metrics`, `Visualizer` | Utilitaires (I/O, preprocessing, postprocessing, métriques, visualisation) |
| `utils` | `ISPY2Extractor` | Extraction et organisation BIDS du dataset ISPY2 |

## Tests

```bash
pytest tests/
```

## Documentation complète

La documentation détaillée de chaque module (tutoriel, guide, référence) se trouve dans [`src/README.md`](src/README.md).
