# Changelog

Toutes les modifications notables de ce package sont documentées ici.
Le versionnage suit [Semantic Versioning](https://semver.org/lang/fr/).

## [0.2.1] - 2026-07-26

### Modifié
- Enrichissement de toutes les docstrings (`MathMorphology`,
  `MorphologyEnhancement`, `MorphologyCleaning`, `MorphologySkeleton`,
  `DistanceTransform`, `WatershedSegmentation`, `MorphologyShape`,
  `MorphologyStats`) avec, pour chaque méthode : contexte historique
  (inventeur, année, laboratoire), section "Quand l'utiliser" (cas
  d'application concrets), et liens de référence pour approfondir.
  `help(Classe.methode)` affiche désormais une fiche complète, façon
  documentation de recherche.

## [0.2.0] - 2026-07-26

### Ajouté
- `MathMorphology.gradient_morphologique()` : détection de contours par
  différence dilatation/érosion.
- `MorphologyEnhancement` (nouveau module) :
  - `top_hat_blanc()`, `top_hat_noir()` — rehaussement de contraste.
- `MorphologyCleaning` (nouveau module) :
  - `supprimer_petits_objets()`, `supprimer_petits_trous()`
  - `etiqueter_composantes()`, `garder_plus_grande_composante()`
- `MorphologySkeleton` (nouveau module) :
  - `squelettiser()` (2D/3D), `axe_median()` (2D).
- `DistanceTransform` (nouveau module) :
  - `transformee_distance()`.
- `WatershedSegmentation` (nouveau module) :
  - `segmenter()` — watershed contrôlé par marqueurs.
- `MorphologyShape` (nouveau module) :
  - `enveloppe_convexe()`, `indice_convexite()`.
- 20 nouveaux tests unitaires (38 au total).

### Corrigé
- Utilisation de `max_size` (au lieu de `min_size`/`area_threshold`,
  dépréciés) pour `remove_small_objects`/`remove_small_holes`,
  compatibilité scikit-image ≥ 0.26.

## [0.1.0] - 2026-07-26

### Ajouté
- `NiftiIO` : chargement / sauvegarde de fichiers NIfTI (`.nii`, `.nii.gz`)
  avec conservation de l'affine et du header.
- `MathMorphology` :
  - `dilatation`, `erosion`
  - `ouverture`, `fermeture`
  - `reconstruction`, `erosion_geodesique`
  - Support des formes `ball`, `cube`, `disk`, `square`.
- `MorphologyStats` :
  - `resume()` (volumes, delta, composantes connexes avant/après)
  - `histogramme_intensites()`
  - `histogramme_volume_comparatif()`
  - `afficher_coupes()`
  - `rapport_complet()`
  - Option `save_fig` pour sauvegarder les figures en `.png`.
- Structure de package installable (`pyproject.toml`, compatible `pip`/`uv`).
- Suite de tests unitaires (`pytest`) pour les 3 modules.
