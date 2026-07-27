"""
mathmorpho.cleaning
=====================

Nettoyage post-segmentation basé sur la taille et la connectivité des
composantes, plutôt que sur la forme d'un élément structurant. Particulièrement
adapté au nettoyage de sorties de modèles de deep learning en segmentation
tumorale.
"""

from __future__ import annotations

import numpy as np
from skimage.measure import label as _skimage_label
from skimage.measure import regionprops as _skimage_regionprops
from skimage.morphology import remove_small_holes as _skimage_remove_small_holes
from skimage.morphology import remove_small_objects as _skimage_remove_small_objects


class MorphologyCleaning:
    """
    Nettoyage d'un masque binaire par taille et connectivité des
    composantes connexes.

    Parameters
    ----------
    image : np.ndarray
        Masque binaire (0/1 ou booléen).

    Examples
    --------
    >>> from mathmorpho import MorphologyCleaning
    >>> nettoyeur = MorphologyCleaning(masque)
    >>> propre = nettoyeur.supprimer_petits_objets(taille_min=50)
    """

    def __init__(self, image: np.ndarray):
        self.image = np.asarray(image)

    def set_image(self, image: np.ndarray) -> None:
        """Remplace l'image de travail sans recréer l'objet."""
        self.image = np.asarray(image)

    def supprimer_petits_objets(self, taille_min: int = 64, connectivite: int = 1) -> np.ndarray:
        """
        Supprime les composantes connexes claires de taille (en voxels)
        inférieure à `taille_min`.

        Contrairement à l'ouverture morphologique, ce filtrage se base
        directement sur la taille réelle des objets et non sur la forme
        d'un élément structurant : plus adapté pour éliminer les faux
        positifs de petite taille issus d'une segmentation automatique.

        Historique
        ----------
        Dérive des algorithmes classiques d'analyse de composantes
        connexes (Rosenfeld & Pfaltz, 1966), popularisée comme outil de
        nettoyage standard dans les bibliothèques modernes de traitement
        d'images (ex : scikit-image) pour le post-traitement de masques
        de segmentation.

        Quand l'utiliser
        -----------------
        - Nettoyer une sortie de modèle de deep learning en supprimant
          les faux positifs isolés, sans dépendre d'un élément
          structurant fixe (contrairement à l'ouverture).
        - Filtrer des objets par un critère de taille clinique connu
          (ex : ignorer toute lésion détectée de moins de X voxels).

        Parameters
        ----------
        taille_min : int, default=64
            Taille minimale (en voxels) en dessous de laquelle un objet
            est supprimé.
        connectivite : int, default=1
            Connectivité utilisée pour définir les composantes (1 = voisins
            directs / 6-connectivité en 3D, jusqu'à `image.ndim` = 26-connectivité
            en 3D).

        Returns
        -------
        np.ndarray
            Masque binaire nettoyé (même type que l'entrée, booléen).

        Pour aller plus loin
        ---------------------
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_objects

        Examples
        --------
        >>> propre = nettoyeur.supprimer_petits_objets(taille_min=100)
        """
        masque_binaire = self.image.astype(bool)
        return _skimage_remove_small_objects(
            masque_binaire, max_size=taille_min - 1, connectivity=connectivite
        )

    def supprimer_petits_trous(self, taille_min: int = 64, connectivite: int = 1) -> np.ndarray:
        """
        Comble les trous (régions de fond entourées par l'objet) de taille
        (en voxels) inférieure à `taille_min`.

        Historique
        ----------
        Opération duale de `supprimer_petits_objets`, basée sur les mêmes
        principes d'analyse de composantes connexes (Rosenfeld & Pfaltz,
        1966), appliqués ici aux trous internes plutôt qu'aux objets.

        Quand l'utiliser
        -----------------
        - Combler des trous internes dans une segmentation tumorale (ex :
          un vaisseau non rehaussé au centre d'une tumeur qui casse la
          continuité du masque).
        - Nettoyer une segmentation par seuillage qui laisse des pixels
          isolés non détectés à l'intérieur d'une région homogène.

        Parameters
        ----------
        taille_min : int, default=64
            Taille maximale (en voxels) des trous à combler.
        connectivite : int, default=1
            Connectivité utilisée pour définir les trous.

        Returns
        -------
        np.ndarray
            Masque binaire avec les petits trous comblés.

        Pour aller plus loin
        ---------------------
        - https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.remove_small_holes

        Examples
        --------
        >>> comble = nettoyeur.supprimer_petits_trous(taille_min=100)
        """
        masque_binaire = self.image.astype(bool)
        return _skimage_remove_small_holes(
            masque_binaire, max_size=taille_min, connectivity=connectivite
        )

    def etiqueter_composantes(self, connectivite: int = 1):
        """
        Étiquette individuellement chaque composante connexe et calcule
        leurs propriétés (volume, centroïde, boîte englobante).

        Historique
        ----------
        Basée sur l'algorithme classique d'étiquetage de composantes
        connexes (connected-component labeling), formalisé par Azriel
        Rosenfeld et John Pfaltz en 1966, pierre angulaire de l'analyse
        d'images binaires.

        Quand l'utiliser
        -----------------
        - Compter le nombre de lésions distinctes dans une segmentation.
        - Extraire des caractéristiques quantitatives (volume, position)
          par lésion individuelle, pour un suivi longitudinal ou une
          analyse statistique.
        - Étape préalable à `garder_plus_grande_composante` ou à un
          filtrage personnalisé par propriété.

        Parameters
        ----------
        connectivite : int, default=1
            Connectivité utilisée pour définir les composantes.

        Returns
        -------
        labels : np.ndarray
            Image où chaque composante connexe porte un entier unique
            (0 = fond).
        nb_composantes : int
            Nombre total de composantes détectées.
        proprietes : list of dict
            Liste de dictionnaires (un par composante) contenant au
            minimum ``label``, ``volume`` (nombre de voxels) et
            ``centroide``.

        Pour aller plus loin
        ---------------------
        - https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label
        - https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops

        Examples
        --------
        >>> labels, nb, props = nettoyeur.etiqueter_composantes()
        >>> volumes = [p["volume"] for p in props]
        """
        masque_binaire = self.image.astype(bool)
        labels, nb_composantes = _skimage_label(
            masque_binaire, connectivity=connectivite, return_num=True
        )

        proprietes = []
        for region in _skimage_regionprops(labels):
            proprietes.append(
                {
                    "label": region.label,
                    "volume": region.area,
                    "centroide": region.centroid,
                    "bounding_box": region.bbox,
                }
            )

        return labels, nb_composantes, proprietes

    def garder_plus_grande_composante(self, connectivite: int = 1) -> np.ndarray:
        """
        Ne conserve que la plus grande composante connexe du masque (utile
        pour isoler la tumeur principale et écarter les faux positifs
        isolés restants).

        Historique
        ----------
        Application directe de l'étiquetage de composantes connexes
        (Rosenfeld & Pfaltz, 1966), très utilisée en pratique clinique
        pour isoler automatiquement la lésion principale d'intérêt.

        Quand l'utiliser
        -----------------
        - Isoler automatiquement la tumeur principale lorsque la
          segmentation produit plusieurs régions disjointes.
        - Dernière étape de nettoyage avant export du masque final,
          quand on sait qu'une seule lésion d'intérêt est attendue.

        Parameters
        ----------
        connectivite : int, default=1
            Connectivité utilisée pour définir les composantes.

        Returns
        -------
        np.ndarray
            Masque binaire ne contenant que la plus grande composante.

        Pour aller plus loin
        ---------------------
        - https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label

        Examples
        --------
        >>> tumeur_principale = nettoyeur.garder_plus_grande_composante()
        """
        labels, nb_composantes, proprietes = self.etiqueter_composantes(connectivite)

        if nb_composantes == 0:
            return np.zeros_like(self.image, dtype=bool)

        plus_grande = max(proprietes, key=lambda p: p["volume"])
        return labels == plus_grande["label"]
