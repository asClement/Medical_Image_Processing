"""
mathmorpho.skeleton
=====================

Squelettisation topologique d'objets binaires (2D ou 3D), utile pour
l'analyse de structures allongées comme les vaisseaux tumoraux.
"""

from __future__ import annotations

import numpy as np

from skimage.morphology import skeletonize as _skimage_skeletonize
from skimage.morphology import medial_axis as _skimage_medial_axis


class MorphologySkeleton:
    """
    Squelettisation d'un objet binaire : réduit l'objet à un ensemble de
    voxels d'épaisseur 1 tout en conservant sa topologie et sa connectivité.

    Parameters
    ----------
    image : np.ndarray
        Masque binaire 2D ou 3D.

    Examples
    --------
    >>> from mathmorpho import MorphologySkeleton
    >>> squelette_obj = MorphologySkeleton(masque_vaisseaux)
    >>> squelette = squelette_obj.squelettiser()
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def squelettiser(self) -> np.ndarray:
        """
        Calcule le squelette topologique de l'objet binaire (fonctionne en
        2D comme en 3D).

        Historique
        ----------
        Concept introduit par Harry Blum en 1967 sous le nom d'axe médian
        ("medial axis transform"), puis formalisé dans le cadre de la
        morphologie mathématique par Serra dans les années 1980 comme
        squelette morphologique (érosions successives avec conservation
        des points de rupture topologique).

        Quand l'utiliser
        -----------------
        - Analyser la forme et la longueur de structures allongées, comme
          les vaisseaux tumoraux (angiogenèse).
        - Réduire une structure complexe à sa topologie essentielle pour
          en simplifier l'analyse quantitative (nombre de branches,
          longueur totale, points de bifurcation).
        - Prétraitement pour des mesures de tortuosité vasculaire.

        Returns
        -------
        np.ndarray
            Masque binaire du squelette (même forme que l'image d'entrée).

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Topological_skeleton
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize

        Examples
        --------
        >>> squelette = squelette_obj.squelettiser()
        """
        masque_binaire = self.image.astype(bool)
        return _skimage_skeletonize(masque_binaire)

    def axe_median(self):
        """
        Calcule l'axe médian (medial axis) en 2D, avec en prime la
        transformée en distance associée à chaque point du squelette.
        Contrairement à `squelettiser`, cette méthode est limitée aux
        images 2D.

        Historique
        ----------
        Concept original de Harry Blum (1967), défini comme le lieu des
        centres des cercles maximaux inscrits dans l'objet — chaque point
        de l'axe médian est équidistant d'au moins deux points du bord.

        Quand l'utiliser
        -----------------
        - Obtenir, en plus du squelette, la distance au bord en chaque
          point (utile pour estimer l'épaisseur locale d'une structure,
          ex : diamètre d'un vaisseau le long de son axe).
        - Analyse fine de coupes 2D individuelles extraites d'un volume 3D.

        Returns
        -------
        squelette : np.ndarray
            Masque binaire de l'axe médian.
        distance : np.ndarray
            Distance de chaque point du squelette au bord de l'objet le
            plus proche.

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Medial_axis
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.medial_axis

        Examples
        --------
        >>> squelette, distance = squelette_obj.axe_median()
        """
        if self.image.ndim != 2:
            raise ValueError("axe_median ne fonctionne qu'en 2D ; utiliser squelettiser() en 3D.")
        masque_binaire = self.image.astype(bool)
        squelette, distance = _skimage_medial_axis(masque_binaire, return_distance=True)
        return squelette, distance
