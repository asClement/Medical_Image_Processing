"""
mathmorpho.distance
=====================

Transformée en distance euclidienne, brique fondamentale pour générer des
marqueurs avant une segmentation par watershed.
"""

from __future__ import annotations

import numpy as np

from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt


class DistanceTransform:
    """
    Calcule la transformée en distance euclidienne d'un masque binaire :
    pour chaque voxel de l'objet, la distance au voxel de fond le plus
    proche.

    Parameters
    ----------
    image : np.ndarray
        Masque binaire 2D ou 3D.

    Examples
    --------
    >>> from mathmorpho import DistanceTransform
    >>> dt = DistanceTransform(masque)
    >>> carte_distance = dt.transformee_distance()
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def transformee_distance(self, sampling=None) -> np.ndarray:
        """
        Calcule la transformée en distance euclidienne.

        Historique
        ----------
        Concept introduit par Azriel Rosenfeld et John Pfaltz en 1966
        ("Sequential operations in digital picture processing"), devenu
        un outil fondamental du traitement d'image, notamment comme brique
        de base de la segmentation par watershed (Beucher & Lantuéjoul,
        1979).

        Quand l'utiliser
        -----------------
        - Générer des marqueurs internes avant une segmentation par
          watershed (les maxima locaux de la carte de distance indiquent
          les centres probables des objets).
        - Estimer l'épaisseur locale d'une structure (distance au bord le
          plus proche en chaque point).
        - Pondérer une analyse en fonction de la proximité au bord d'une
          lésion.

        Parameters
        ----------
        sampling : tuple of float, optional
            Espacement des voxels selon chaque dimension (ex : le
            `zooms`/voxel spacing d'un fichier NIfTI), pour une distance
            exprimée dans l'unité physique réelle plutôt qu'en voxels.
            Si None, un espacement uniforme de 1 est utilisé.

        Returns
        -------
        np.ndarray
            Carte de distance, même forme que l'image d'entrée.

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Distance_transform
        - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html

        Examples
        --------
        >>> carte = dt.transformee_distance(sampling=(1.0, 1.0, 1.5))
        """
        masque_binaire = self.image.astype(bool)
        return _scipy_distance_transform_edt(masque_binaire, sampling=sampling)
