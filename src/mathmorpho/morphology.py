"""
mathmorpho.morphology
======================

Opérations de morphologie mathématique 3D pour le prétraitement d'images
IRM (ou tout volume numpy), destinées en particulier à l'imagerie tumorale :
dilatation, érosion, ouverture, fermeture, reconstruction géodésique.

Références scientifiques
-------------------------
- Matheron G., Serra J. — École des Mines de Paris / Centre de Morphologie
  Mathématique (1960-1970).
- Vincent L. (1993) — Morphological grayscale reconstruction in image
  analysis: applications and efficient algorithms.
"""

from __future__ import annotations

import numpy as np

from skimage.morphology import (
    ball,
    disk,
    footprint_rectangle,
    erosion as _skimage_erosion,
    dilation as _skimage_dilation,
    opening as _skimage_opening,
    closing as _skimage_closing,
    reconstruction as _skimage_reconstruction,
)


class MathMorphology:
    """
    Implémentation des opérations de morphologie mathématique 3D :
    dilatation, érosion, ouverture, fermeture et reconstruction géodésique.

    Fonctionne aussi bien sur :

    - des masques binaires (0/1) issus d'une segmentation,
    - des images en niveaux de gris (grayscale), ex : IRM brute ou carte
      de probabilité issue d'un modèle de deep learning.

    Le type d'image (binaire ou grayscale) est détecté automatiquement.

    Parameters
    ----------
    image : np.ndarray
        Volume (généralement 3D) chargé depuis un fichier NIfTI.

    Examples
    --------
    >>> from mathmorpho import MathMorphology
    >>> morpho = MathMorphology(volume)
    >>> masque_propre = morpho.ouverture(forme="ball", rayon=1)
    """

    FORMES_DISPONIBLES_3D = {"ball", "cube"}
    FORMES_DISPONIBLES_2D = {"disk", "square"}

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    # ------------------------------------------------------------------
    # Gestion de l'image
    # ------------------------------------------------------------------
    def set_image(self, image: np.ndarray) -> None:
        """
        Remplace l'image de travail sans recréer l'objet.

        Parameters
        ----------
        image : np.ndarray
            Nouvelle image à utiliser pour les prochains appels de méthode.

        Examples
        --------
        >>> morpho.set_image(nouveau_volume)
        """
        self.image = np.asarray(image)

    def _est_binaire(self, image: np.ndarray) -> bool:
        """
        Détecte si une image est binaire (masque de segmentation) ou
        en niveaux de gris (au plus 2 valeurs uniques).
        """
        valeurs_uniques = np.unique(image)
        return valeurs_uniques.size <= 2

    # ------------------------------------------------------------------
    # Élément structurant
    # ------------------------------------------------------------------
    def _get_element_structurant(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Construit l'élément structurant utilisé par les opérations
        morphologiques.

        Parameters
        ----------
        forme : str
            'ball' (sphère 3D), 'cube' (3D), 'disk' (2D), 'square' (2D).
        rayon : int
            Rayon (ou demi-côté) de l'élément structurant, en voxels.

        Returns
        -------
        np.ndarray
            L'élément structurant correspondant.
        """
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
            raise ValueError(
                f"Forme d'élément structurant inconnue : '{forme}'. "
                f"Choix possibles : {self.FORMES_DISPONIBLES_3D | self.FORMES_DISPONIBLES_2D}"
            )

    # ------------------------------------------------------------------
    # 6.1 Dilatation et Érosion
    # ------------------------------------------------------------------
    def dilatation(self, iterations: int = 1, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Dilatation morphologique : D(A, B) = { z : (B_z) ∩ A ≠ ∅ }.

        Agrandit les régions claires (objets) de l'image. Utile par exemple
        pour étendre une région tumorale segmentée afin d'inclure une marge
        péritumorale (œdème).

        Historique
        ----------
        Inventée par Georges Matheron et Jean Serra à l'École des Mines de
        Paris (Centre de Morphologie Mathématique), dans les années
        1960-1970, à l'origine pour l'analyse de pétrographie et de
        matériaux poreux, puis étendue à l'analyse d'images en général.

        Quand l'utiliser
        -----------------
        - Étendre une région segmentée (ex : ajouter une marge de sécurité
          autour d'une tumeur pour couvrir l'œdème péritumoral).
        - Combler visuellement de petits espaces entre objets proches
          avant une fermeture.
        - Reconnecter des fragments d'une même structure très proches.

        Parameters
        ----------
        iterations : int, default=1
            Nombre de fois où la dilatation est appliquée successivement.
        forme : str, default='ball'
            Forme de l'élément structurant ('ball', 'cube', 'disk', 'square').
        rayon : int, default=1
            Rayon de l'élément structurant.

        Returns
        -------
        np.ndarray
            Image dilatée.

        Pour aller plus loin
        ---------------------
        - https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT3/node3.html
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.dilation

        Examples
        --------
        >>> dilate = morpho.dilatation(iterations=2, forme="ball", rayon=1)
        """
        selem = self._get_element_structurant(forme, rayon)
        resultat = self.image.copy()
        for _ in range(iterations):
            resultat = _skimage_dilation(resultat, footprint=selem)
        return resultat

    def erosion(self, iterations: int = 1, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Érosion morphologique : E(A, B) = { z : (B_z) ⊆ A }.

        Rétrécit les régions claires (objets) de l'image. Utile pour
        éliminer les petits artefacts de segmentation ou composantes
        isolées de faible taille.

        Historique
        ----------
        Inventée par Georges Matheron et Jean Serra à l'École des Mines de
        Paris (Centre de Morphologie Mathématique), dans les années
        1960-1970, en même temps que la dilatation, comme opération duale.

        Quand l'utiliser
        -----------------
        - Éliminer de petits artefacts de segmentation isolés (bruit).
        - Séparer deux objets légèrement connectés avant un étiquetage
          des composantes connexes.
        - Générer un marqueur "sûr" (sous-ensemble certain d'un objet)
          pour une reconstruction géodésique ou un watershed.

        Parameters
        ----------
        iterations : int, default=1
            Nombre de fois où l'érosion est appliquée successivement.
        forme : str, default='ball'
            Forme de l'élément structurant ('ball', 'cube', 'disk', 'square').
        rayon : int, default=1
            Rayon de l'élément structurant.

        Returns
        -------
        np.ndarray
            Image érodée.

        Pour aller plus loin
        ---------------------
        - https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT3/node3.html
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.erosion

        Examples
        --------
        >>> erode = morpho.erosion(iterations=1, forme="ball", rayon=1)
        """
        selem = self._get_element_structurant(forme, rayon)
        resultat = self.image.copy()
        for _ in range(iterations):
            resultat = _skimage_erosion(resultat, footprint=selem)
        return resultat

    # ------------------------------------------------------------------
    # 6.2 Ouverture et Fermeture
    # ------------------------------------------------------------------
    def ouverture(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Ouverture morphologique : A ∘ B = (A ⊖ B) ⊕ B.

        Opération anti-extensive et idempotente. Supprime les petites
        structures claires (bruit, fausses détections de petite taille)
        et lisse les contours en creux.

        Historique
        ----------
        Introduite par Georges Matheron et Jean Serra (Centre de
        Morphologie Mathématique, École des Mines de Paris, 1960-1970)
        comme composition de l'érosion et de la dilatation.

        Quand l'utiliser
        -----------------
        - Nettoyer une segmentation automatique de ses faux positifs de
          petite taille, sans trop déformer les objets principaux.
        - Lisser les contours irréguliers en creux d'une lésion.
        - Prétraitement standard après une segmentation par deep learning.

        Parameters
        ----------
        forme : str, default='ball'
            Forme de l'élément structurant.
        rayon : int, default=1
            Rayon de l'élément structurant (typiquement 1 à 3 voxels).

        Returns
        -------
        np.ndarray
            Image après ouverture morphologique.

        Pour aller plus loin
        ---------------------
        - https://wjarr.com/sites/default/files/fulltext_pdf/WJARR-2022-0576.pdf
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.opening

        Examples
        --------
        >>> propre = morpho.ouverture(forme="ball", rayon=1)
        """
        selem = self._get_element_structurant(forme, rayon)
        return _skimage_opening(self.image, footprint=selem)

    def fermeture(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Fermeture morphologique : A • B = (A ⊕ B) ⊖ B.

        Opération extensive et idempotente. Comble les petits trous et
        gaps, utile pour reconnecter des régions tumorales fragmentées
        par le bruit ou les artefacts de mouvement.

        Historique
        ----------
        Introduite par Georges Matheron et Jean Serra (Centre de
        Morphologie Mathématique, École des Mines de Paris, 1960-1970),
        opération duale de l'ouverture.

        Quand l'utiliser
        -----------------
        - Reconnecter des fragments d'une même région tumorale séparés
          par du bruit ou un artefact de mouvement.
        - Combler de petits trous internes dans un masque de
          segmentation (ex : un vaisseau non détecté au centre d'une
          tumeur).
        - Lisser les contours irréguliers en bosse d'une lésion.

        Parameters
        ----------
        forme : str, default='ball'
            Forme de l'élément structurant.
        rayon : int, default=1
            Rayon de l'élément structurant (typiquement 1 à 3 voxels).

        Returns
        -------
        np.ndarray
            Image après fermeture morphologique.

        Pour aller plus loin
        ---------------------
        - https://wjarr.com/sites/default/files/fulltext_pdf/WJARR-2022-0576.pdf
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.closing

        Examples
        --------
        >>> comble = morpho.fermeture(forme="ball", rayon=1)
        """
        selem = self._get_element_structurant(forme, rayon)
        return _skimage_closing(self.image, footprint=selem)

    # ------------------------------------------------------------------
    # 6.3 Reconstruction morphologique
    # ------------------------------------------------------------------
    def reconstruction(
        self,
        marqueur: np.ndarray,
        masque: np.ndarray = None,
        methode: str = "dilatation",
    ) -> np.ndarray:
        """
        Reconstruction morphologique géodésique (Vincent, 1993).

        ρ_I(f) = sup{ k : T_k(f) ≤ I }, où T_k est la dilatation
        géodésique itérée.

        Reconstruit les objets du masque connectés aux marqueurs, sans en
        créer de nouveaux.

        Historique
        ----------
        Formalisée par Luc Vincent en 1993 ("Morphological grayscale
        reconstruction in image analysis: applications and efficient
        algorithms"), sur la base des travaux fondateurs de Jean Serra et
        Georges Matheron sur les transformations géodésiques.

        Quand l'utiliser
        -----------------
        - Extraire uniquement les structures d'une image connectées à un
          marqueur connu (ex : garder la tumeur reliée à un point
          d'intérêt, en écartant toute structure non connectée).
        - Prétraitement classique avant une segmentation par watershed
          (les marqueurs définissent les régions de départ).
        - Filtrage plus fin qu'une ouverture/fermeture car il préserve la
          forme exacte des objets conservés.

        Parameters
        ----------
        marqueur : np.ndarray
            Image marqueur (souvent une érosion de l'image d'origine).
        masque : np.ndarray, optional
            Image masque définissant la contrainte de reconstruction.
            Si None, self.image est utilisée comme masque.
        methode : str, default='dilatation'
            'dilatation' (marqueur ⊆ masque requis) ou
            'erosion' (marqueur ⊇ masque requis).

        Returns
        -------
        np.ndarray
            Image reconstruite.

        Pour aller plus loin
        ---------------------
        - https://cseweb.ucsd.edu/classes/fa23/cse166-a/lec13.pdf
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.reconstruction

        Examples
        --------
        >>> marqueur = morpho.erosion(rayon=2)
        >>> reconstruit = morpho.reconstruction(marqueur, methode="dilatation")
        """
        if masque is None:
            masque = self.image

        methode = methode.lower()
        if methode not in ("dilatation", "erosion"):
            raise ValueError("methode doit être 'dilatation' ou 'erosion'.")

        # skimage attend les mots-clés anglais 'dilation'/'erosion'
        methode_skimage = "dilation" if methode == "dilatation" else "erosion"

        return _skimage_reconstruction(marqueur, masque, method=methode_skimage)

    def gradient_morphologique(self, forme: str = "ball", rayon: int = 1) -> np.ndarray:
        """
        Gradient morphologique : Grad(f) = (f ⊕ b) - (f ⊖ b).

        Différence entre la dilatation et l'érosion. Met en évidence les
        frontières/contours des objets. Sert classiquement de prétraitement
        avant une segmentation par watershed.

        Historique
        ----------
        Concept dérivé directement des opérations de base de Matheron et
        Serra (1960-1970) ; popularisé comme outil de détection de
        contours par l'école de morphologie mathématique française dans
        les années 1970-1980 (notamment pour les applications en
        segmentation par ligne de partage des eaux).

        Quand l'utiliser
        -----------------
        - Détecter les contours d'une tumeur ou d'une structure anatomique
          avant une segmentation par watershed.
        - Visualiser la frontière exacte d'un objet segmenté.
        - Alternative morphologique aux filtres de gradient classiques
          (Sobel, Prewitt) quand on travaille déjà avec des éléments
          structurants définis.

        Parameters
        ----------
        forme : str, default='ball'
            Forme de l'élément structurant.
        rayon : int, default=1
            Rayon de l'élément structurant.

        Returns
        -------
        np.ndarray
            Image du gradient morphologique (contours).

        Pour aller plus loin
        ---------------------
        - https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/OWENS/LECT3/node3.html
        - https://en.wikipedia.org/wiki/Morphological_gradient

        Examples
        --------
        >>> contours = morpho.gradient_morphologique(forme="ball", rayon=1)
        """
        selem = self._get_element_structurant(forme, rayon)
        dilate = _skimage_dilation(self.image, footprint=selem)
        erode = _skimage_erosion(self.image, footprint=selem)
        return dilate.astype(np.float64) - erode.astype(np.float64)

    def erosion_geodesique(self, marqueur: np.ndarray, masque: np.ndarray = None) -> np.ndarray:
        """
        Cas particulier de reconstruction morphologique par érosion
        géodésique, utile comme prétraitement au watershed.

        Historique
        ----------
        Cas particulier de la reconstruction morphologique de Luc Vincent
        (1993), basée sur les transformations géodésiques itérées définies
        par Serra et Matheron.

        Quand l'utiliser
        -----------------
        - Générer des marqueurs propres avant un watershed, en ne
          conservant que les parties du masque réellement connectées au
          marqueur.
        - Filtrer une image en respectant une contrainte de connectivité
          plus stricte qu'une simple érosion.

        Parameters
        ----------
        marqueur : np.ndarray
            Image marqueur (doit être ≥ masque point par point).
        masque : np.ndarray, optional
            Image masque. Si None, self.image est utilisée.

        Returns
        -------
        np.ndarray
            Image reconstruite par érosion géodésique.

        Pour aller plus loin
        ---------------------
        - https://cseweb.ucsd.edu/classes/fa23/cse166-a/lec13.pdf

        Examples
        --------
        >>> reconstruit = morpho.erosion_geodesique(marqueur)
        """
        return self.reconstruction(marqueur, masque, methode="erosion")
