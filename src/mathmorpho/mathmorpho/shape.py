"""
mathmorpho.shape
==================

Analyse de forme par enveloppe convexe. Utile en oncologie pour quantifier
l'irrégularité d'une lésion (l'écart entre la forme réelle et sa version
convexe peut être un indicateur d'infiltration).
"""

from __future__ import annotations

import numpy as np

from skimage.morphology import convex_hull_image as _skimage_convex_hull_image


class MorphologyShape:
    """
    Analyse de forme d'un objet binaire par enveloppe convexe.

    Parameters
    ----------
    image : np.ndarray
        Masque binaire 2D ou 3D.

    Examples
    --------
    >>> from mathmorpho import MorphologyShape
    >>> forme = MorphologyShape(masque_tumeur)
    >>> enveloppe = forme.enveloppe_convexe()
    >>> indice = forme.indice_convexite()
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def enveloppe_convexe(self) -> np.ndarray:
        """
        Calcule la plus petite forme convexe englobant l'objet binaire.

        Historique
        ----------
        Concept issu de la géométrie algorithmique classique (Graham,
        1972 ; Preparata & Shamos, 1985), intégré comme opération standard
        de la morphologie mathématique pour l'analyse de forme (Serra,
        "Image Analysis and Mathematical Morphology", 1982).

        Quand l'utiliser
        -----------------
        - Quantifier la régularité/irrégularité de la forme d'une lésion
          en la comparant à sa version convexe (voir `indice_convexite`).
        - Obtenir une région englobante "raisonnable" pour définir une
          zone d'intérêt élargie autour d'un objet irrégulier.

        Returns
        -------
        np.ndarray
            Masque binaire de l'enveloppe convexe (même forme que l'image
            d'entrée).

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Convex_hull
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.convex_hull_image

        Examples
        --------
        >>> enveloppe = forme.enveloppe_convexe()
        """
        masque_binaire = self.image.astype(bool)
        return _skimage_convex_hull_image(masque_binaire)

    def indice_convexite(self) -> float:
        """
        Calcule l'indice de convexité : rapport entre le volume de l'objet
        et le volume de son enveloppe convexe.

        Un indice proche de 1 indique une forme régulière (proche de
        convexe), un indice plus faible indique une forme irrégulière ou
        infiltrante — un marqueur potentiellement pertinent en analyse de
        lésions tumorales.

        Historique
        ----------
        Dérivé directement du concept d'enveloppe convexe (Graham, 1972),
        utilisé en radiomique moderne comme descripteur de forme simple
        et interprétable (les indices de forme basés sur la convexité sont
        couramment étudiés dans la littérature de radiomique en oncologie
        depuis les années 2010).

        Quand l'utiliser
        -----------------
        - Quantifier objectivement l'irrégularité d'une tumeur : une
          forme très infiltrante ou spiculée aura un indice bas, une forme
          arrondie/régulière un indice proche de 1.
        - Générer une caractéristique (feature) de forme pour un modèle
          de radiomique ou une analyse pronostique.

        Returns
        -------
        float
            Indice de convexité, compris entre 0 et 1.

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Convex_hull
        - https://pyradiomics.readthedocs.io/en/latest/features.html#module-radiomics.shape

        Examples
        --------
        >>> indice = forme.indice_convexite()
        """
        masque_binaire = self.image.astype(bool)
        volume_objet = masque_binaire.sum()
        if volume_objet == 0:
            return float("nan")

        enveloppe = self.enveloppe_convexe()
        volume_enveloppe = enveloppe.sum()

        return float(volume_objet / volume_enveloppe)
