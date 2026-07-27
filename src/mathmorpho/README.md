# mathmorpho

Package Python de **morphologie mathématique 3D** pour le prétraitement
d'images IRM au format **NIfTI** (`.nii` / `.nii.gz`), orienté imagerie
tumorale.

Implémente les opérations classiques (Matheron & Serra, 1960-1970 ;
Vincent, 1993) : dilatation, érosion, ouverture, fermeture, reconstruction
géodésique — ainsi qu'un module de statistiques comparatives avant/après
transformation.

## Installation

Avec `uv` (recommandé) :

```bash
uv pip install -e .
```

Avec `pip` :

```bash
pip install -e .
```

Pour les dépendances de développement (tests) :

```bash
uv pip install -e ".[dev]"
```

## Démarrage rapide

```python
from mathmorpho import NiftiIO, MathMorphology, MorphologyStats

# 1. Charger un volume NIfTI (masque de segmentation ou IRM)
volume, affine, header = NiftiIO.charger_nifti("segmentation.nii.gz")

# 2. Appliquer une transformation morphologique
morpho = MathMorphology(volume)
volume_propre = morpho.ouverture(forme="ball", rayon=1)

# 3. Générer des statistiques comparatives avant / après
stats = MorphologyStats(volume, volume_propre, save_fig=True, dossier_sortie="figures")
stats.rapport_complet()

# 4. Sauvegarder le résultat en conservant les métadonnées d'origine
NiftiIO.sauvegarder_nifti(volume_propre, affine, "segmentation_clean.nii.gz", header)
```

## Documentation des méthodes

Comme avec `numpy` ou `scikit-learn`, la documentation de chaque classe et
méthode est accessible directement via `help()`. Chaque méthode inclut :

- la **formule mathématique** et une description fonctionnelle,
- son **historique** (qui l'a inventée, quand, dans quel laboratoire),
- une section **"Quand l'utiliser"** avec des cas d'usage concrets en
  imagerie médicale,
- des **liens de référence** pour approfondir,
- un exemple d'usage.

```python
from mathmorpho import MathMorphology
help(MathMorphology)
help(MathMorphology.dilatation)
```

## Classes disponibles

### `NiftiIO`
| Méthode | Description |
|---|---|
| `charger_nifti(chemin)` | Charge un `.nii`/`.nii.gz` → `(array, affine, header)` |
| `sauvegarder_nifti(array, affine, chemin, header=None)` | Sauvegarde un volume en `.nii.gz` |

### `MathMorphology`
| Méthode | Description |
|---|---|
| `dilatation(iterations=1, forme='ball', rayon=1)` | Dilatation morphologique |
| `erosion(iterations=1, forme='ball', rayon=1)` | Érosion morphologique |
| `ouverture(forme='ball', rayon=1)` | Érosion puis dilatation |
| `fermeture(forme='ball', rayon=1)` | Dilatation puis érosion |
| `gradient_morphologique(forme='ball', rayon=1)` | Contours = dilatation - érosion |
| `reconstruction(marqueur, masque=None, methode='dilatation')` | Reconstruction géodésique |
| `erosion_geodesique(marqueur, masque=None)` | Reconstruction par érosion géodésique |
| `set_image(image)` | Change l'image de travail |

Formes disponibles pour l'élément structurant : `'ball'`, `'cube'` (3D),
`'disk'`, `'square'` (2D).

### `MorphologyEnhancement`
| Méthode | Description |
|---|---|
| `top_hat_blanc(forme='ball', rayon=1)` | Extrait les petites structures claires |
| `top_hat_noir(forme='ball', rayon=1)` | Extrait les petites structures sombres |

### `MorphologyCleaning`
| Méthode | Description |
|---|---|
| `supprimer_petits_objets(taille_min=64, connectivite=1)` | Supprime les objets sous un seuil de taille (voxels) |
| `supprimer_petits_trous(taille_min=64, connectivite=1)` | Comble les trous sous un seuil de taille |
| `etiqueter_composantes(connectivite=1)` | Étiquette les composantes + propriétés (volume, centroïde...) |
| `garder_plus_grande_composante(connectivite=1)` | Ne garde que la plus grande composante connexe |

### `MorphologySkeleton`
| Méthode | Description |
|---|---|
| `squelettiser()` | Squelette topologique (2D ou 3D) |
| `axe_median()` | Axe médian + distance au bord (2D uniquement) |

### `DistanceTransform`
| Méthode | Description |
|---|---|
| `transformee_distance(sampling=None)` | Distance euclidienne au bord le plus proche |

### `WatershedSegmentation`
| Méthode | Description |
|---|---|
| `segmenter(marqueurs=None, distance_min=10)` | Watershed contrôlé par marqueurs (sépare des objets accolés) |

### `MorphologyShape`
| Méthode | Description |
|---|---|
| `enveloppe_convexe()` | Plus petite forme convexe englobant l'objet |
| `indice_convexite()` | Ratio volume objet / volume enveloppe convexe (0-1) |

### `MorphologyStats`
| Méthode | Description |
|---|---|
| `resume()` | Dictionnaire de statistiques comparatives (volumes, delta, composantes connexes...) |
| `afficher_resume()` | Affiche le résumé dans la console |
| `histogramme_intensites(bins=50)` | Histogramme comparatif des intensités |
| `histogramme_volume_comparatif()` | Barplot du volume avant/après |
| `afficher_coupes(axe='axial', indice=None)` | Comparaison visuelle de coupes 2D |
| `rapport_complet()` | Génère l'ensemble des statistiques et graphiques |

Toutes les méthodes de visualisation acceptent le paramètre `save_fig`
(défini à l'initialisation de `MorphologyStats`) : si `True`, les figures
sont sauvegardées en `.png` dans `dossier_sortie` au lieu d'être affichées.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```

## Structure du projet

```
mathmorpho/
├── mathmorpho/
│   ├── __init__.py         # API publique
│   ├── io.py                # NiftiIO
│   ├── morphology.py        # MathMorphology
│   ├── enhancement.py       # MorphologyEnhancement (top-hat)
│   ├── cleaning.py          # MorphologyCleaning (petits objets/trous, composantes)
│   ├── skeleton.py          # MorphologySkeleton
│   ├── distance.py          # DistanceTransform
│   ├── segmentation.py      # WatershedSegmentation
│   ├── shape.py              # MorphologyShape (enveloppe convexe)
│   ├── stats.py              # MorphologyStats
│   └── _version.py
├── tests/
├── examples/
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

## Références scientifiques

- Matheron G., Serra J. — École des Mines de Paris / Centre de
  Morphologie Mathématique (1960-1970).
- Vincent L. (1993) — *Morphological grayscale reconstruction in image
  analysis: applications and efficient algorithms*.

## Licence

MIT
