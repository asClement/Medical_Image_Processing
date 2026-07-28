"""
mathmorpho.segmentation
=========================

Segmentation par ligne de partage des eaux (watershed) contrôlée par
marqueurs, construite à partir de la transformée en distance. Utile pour
séparer des tumeurs ou noyaux accolés après une segmentation binaire
initiale.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt
from scipy.ndimage import label as _scipy_label
from skimage.feature import peak_local_max as _skimage_peak_local_max
from skimage.segmentation import watershed as _skimage_watershed


class WatershedSegmentation:
    """
    Segmentation par watershed contrôlée par marqueurs, à partir d'un
    masque binaire. Les marqueurs sont générés automatiquement à partir
    des maxima locaux de la transformée en distance (méthode classique
    pour séparer des objets accolés), ou peuvent être fournis manuellement.

    Parameters
    ----------
    image : np.ndarray
        Masque binaire 2D ou 3D (ex : résultat d'une segmentation
        tumorale à séparer en objets individuels).

    Examples
    --------
    >>> from mathmorpho import WatershedSegmentation
    >>> ws = WatershedSegmentation(masque)
    >>> labels = ws.segmenter(distance_min=5)
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def segmenter(self, marqueurs: np.ndarray = None, distance_min: int = 10) -> np.ndarray:
        """
        Applique une segmentation watershed contrôlée par marqueurs sur le
        masque binaire.

        Historique
        ----------
        Le concept de ligne de partage des eaux ("watershed") en
        traitement d'image a été introduit par Serge Beucher et Christian
        Lantuéjoul en 1979 dans le cadre du Centre de Morphologie
        Mathématique (École des Mines de Paris). La variante contrôlée par
        marqueurs, utilisée ici, a ensuite été développée par Beucher et
        Meyer dans les années 1990 pour éviter la sur-segmentation
        inhérente à la méthode originale.

        Quand l'utiliser
        -----------------
        - Séparer deux tumeurs (ou noyaux cellulaires) accolées qu'une
          segmentation binaire simple traite comme un seul objet.
        - Individualiser des objets pour un comptage ou une analyse
          statistique par objet après une segmentation grossière.
        - Typiquement utilisé après une transformée en distance
          (`DistanceTransform`), qui fournit les marqueurs automatiques.

        Parameters
        ----------
        marqueurs : np.ndarray, optional
            Image de marqueurs déjà étiquetés (entiers, 0 = pas de
            marqueur). Si None, les marqueurs sont générés automatiquement
            à partir des maxima locaux de la transformée en distance.
        distance_min : int, default=10
            Distance minimale (en voxels) entre deux marqueurs générés
            automatiquement. Une valeur plus grande réduit la
            sur-segmentation, une valeur plus petite permet de séparer des
            objets plus rapprochés.

        Returns
        -------
        np.ndarray
            Image étiquetée (chaque objet séparé porte un entier unique,
            0 = fond).

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Watershed_(image_processing)
        - https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed

        Examples
        --------
        >>> labels = ws.segmenter(distance_min=5)
        >>> nb_objets = labels.max()
        """
        masque_binaire = self.image.astype(bool)
        carte_distance = _scipy_distance_transform_edt(masque_binaire)

        if marqueurs is None:
            coordonnees_pics = _skimage_peak_local_max(
                carte_distance, min_distance=distance_min, labels=masque_binaire
            )
            masque_pics = np.zeros_like(carte_distance, dtype=bool)
            masque_pics[tuple(coordonnees_pics.T)] = True
            marqueurs, _ = _scipy_label(masque_pics)

        return _skimage_watershed(-carte_distance, markers=marqueurs, mask=masque_binaire)
