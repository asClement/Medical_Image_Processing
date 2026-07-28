"""
mathmorpho.enhancement
========================

Rehaussement de contraste par transformées Top-Hat (chapeau haut-de-forme).
Utile pour corriger les inhomogénéités d'intensité (bias field) en IRM et
rehausser les petites structures (vaisseaux, microcalcifications, petites
lésions) plus petites que l'élément structurant.
"""

from __future__ import annotations

import numpy as np

from skimage.morphology import (
    ball,
    disk,
    footprint_rectangle,
    white_tophat as _skimage_white_tophat,
    black_tophat as _skimage_black_tophat,
)


class MorphologyEnhancement:
    """
    Rehaussement de contraste par transformées Top-Hat.

    - Le *White Top-Hat* extrait les petites structures claires plus
      petites que l'élément structurant : WTH = f - (f ∘ b).
    - Le *Black Top-Hat* extrait les petites structures sombres :
      BTH = (f • b) - f.

    Parameters
    ----------
    image : np.ndarray
        Volume ou image à traiter.

    Examples
    --------
    >>> from mathmorpho import MorphologyEnhancement
    >>> enh = MorphologyEnhancement(volume)
    >>> vaisseaux = enh.top_hat_blanc(forme="ball", rayon=2)
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def _get_element_structurant(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        forme = forme.lower()
        if forme == "ball":
            return ball(rayon)
        elif forme == "cube":
            cote = 2 * rayon + 1
            return footprint_rectangle((cote, cote, cote))
        elif forme == "disk":
            return disk(rayon)
        elif forme == "square":
            cote = 2 * rayon + 1
            return footprint_rectangle((cote, cote))
        else:
            raise ValueError(f"Forme d'élément structurant inconnue : '{forme}'.")

    def top_hat_blanc(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        White Top-Hat : WTH = f - (f ∘ b).

        Extrait les petites structures claires (plus petites que
        l'élément structurant) par rapport au fond environnant.

        Historique
        ----------
        Introduite par Fernand Meyer et Jean Serra dans le cadre du Centre
        de Morphologie Mathématique (École des Mines de Paris), à la fin
        des années 1970, comme extension pratique de l'ouverture
        morphologique pour l'analyse de contraste local.

        Quand l'utiliser
        -----------------
        - Corriger une inhomogénéité d'intensité (bias field) en IRM en
          isolant les variations locales de faible échelle.
        - Rehausser le contraste de petites structures claires (vaisseaux
          fins, microcalcifications, petites lésions) noyées dans un fond
          plus sombre et hétérogène.
        - Prétraitement avant seuillage pour la détection de petites
          structures difficiles à segmenter directement.

        Parameters
        ----------
        forme : str, default='ball'
            Forme de l'élément structurant ('ball', 'cube', 'disk', 'square').
        rayon : int, default=1
            Rayon de l'élément structurant. Doit être supérieur à la taille
            des structures que l'on souhaite extraire.

        Returns
        -------
        np.ndarray
            Image des structures claires extraites.

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Top-hat_transform
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.white_tophat

        Examples
        --------
        >>> petites_lesions = enh.top_hat_blanc(forme="ball", rayon=2)
        """
        selem = self._get_element_structurant(forme, rayon)
        return _skimage_white_tophat(self.image, footprint=selem)

    def top_hat_noir(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Black Top-Hat : BTH = (f • b) - f.

        Extrait les petites structures sombres (plus petites que
        l'élément structurant) par rapport au fond environnant.

        Historique
        ----------
        Introduite par Fernand Meyer et Jean Serra (Centre de Morphologie
        Mathématique, fin des années 1970), en tant qu'opération duale du
        White Top-Hat, basée sur la fermeture morphologique.

        Quand l'utiliser
        -----------------
        - Détecter de petites zones sombres (nécrose, hypo-intensités
          focales) noyées dans une région plus claire.
        - Corriger des ombres locales ou artefacts d'assombrissement dans
          une image IRM.
        - Complément du White Top-Hat pour une analyse de contraste
          bidirectionnelle (structures claires ET sombres).

        Parameters
        ----------
        forme : str, default='ball'
            Forme de l'élément structurant ('ball', 'cube', 'disk', 'square').
        rayon : int, default=1
            Rayon de l'élément structurant.

        Returns
        -------
        np.ndarray
            Image des structures sombres extraites.

        Pour aller plus loin
        ---------------------
        - https://en.wikipedia.org/wiki/Top-hat_transform
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.black_tophat

        Examples
        --------
        >>> zones_sombres = enh.top_hat_noir(forme="ball", rayon=2)
        """
        selem = self._get_element_structurant(forme, rayon)
        return _skimage_black_tophat(self.image, footprint=selem)
