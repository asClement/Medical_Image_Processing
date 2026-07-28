"""
mathmorpho
==========

Package Python de morphologie mathématique 3D pour le prétraitement
d'images IRM (NIfTI), orienté imagerie tumorale.

Classes principales
--------------------
NiftiIO
    Chargement et sauvegarde de fichiers NIfTI (.nii / .nii.gz).
MathMorphology
    Dilatation, érosion, ouverture, fermeture, reconstruction géodésique,
    gradient morphologique.
MorphologyEnhancement
    Rehaussement de contraste par transformées Top-Hat (blanc / noir).
MorphologyCleaning
    Nettoyage post-segmentation par taille et connectivité (petits objets,
    petits trous, plus grande composante, étiquetage).
MorphologySkeleton
    Squelettisation topologique (2D/3D) et axe médian (2D).
DistanceTransform
    Transformée en distance euclidienne.
WatershedSegmentation
    Segmentation par watershed contrôlée par marqueurs.
MorphologyShape
    Analyse de forme par enveloppe convexe et indice de convexité.
MorphologyStats
    Statistiques comparatives (volume, composantes connexes, histogrammes)
    entre une image avant et après transformation morphologique.

Exemple rapide
---------------
>>> from mathmorpho import NiftiIO, MathMorphology, MorphologyStats
>>>
>>> volume, affine, header = NiftiIO.charger_nifti("segmentation.nii.gz")
>>> morpho = MathMorphology(volume)
>>> propre = morpho.ouverture(forme="ball", rayon=1)
>>>
>>> stats = MorphologyStats(volume, propre, save_fig=True)
>>> stats.rapport_complet()
>>>
>>> NiftiIO.sauvegarder_nifti(propre, affine, "segmentation_clean.nii.gz", header)

Pour la documentation détaillée de chaque méthode, utiliser `help()` :

>>> help(MathMorphology)
>>> help(MathMorphology.dilatation)
"""

from ._version import __version__
from .io import NiftiIO
from .morphology import MathMorphology
from .enhancement import MorphologyEnhancement
from .cleaning import MorphologyCleaning
from .skeleton import MorphologySkeleton
from .distance import DistanceTransform
from .segmentation import WatershedSegmentation
from .shape import MorphologyShape
from .stats import MorphologyStats

__all__ = [
    "NiftiIO",
    "MathMorphology",
    "MorphologyEnhancement",
    "MorphologyCleaning",
    "MorphologySkeleton",
    "DistanceTransform",
    "WatershedSegmentation",
    "MorphologyShape",
    "MorphologyStats",
    "__version__",
]
