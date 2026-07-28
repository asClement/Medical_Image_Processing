"""
mathmorpho.stats
=================

Statistiques comparatives avant / après une transformation morphologique :
volumes, différences, composantes connexes, histogrammes des intensités,
et visualisation comparative de coupes 2D extraites d'un volume 3D.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label as _scipy_label


class MorphologyStats:
    """
    Calcule et visualise des statistiques comparatives entre une image
    avant transformation et la même image après une opération de
    morphologie mathématique (dilatation, érosion, ouverture, fermeture,
    reconstruction, etc.).

    Parameters
    ----------
    image_avant : np.ndarray
        Image (ou volume 3D) avant transformation.
    image_apres : np.ndarray
        Image (ou volume 3D) après transformation. Doit avoir la même
        forme que `image_avant`.
    save_fig : bool, default=False
        Si True, chaque figure générée est sauvegardée en .png dans
        `dossier_sortie` au lieu d'être seulement affichée.
    dossier_sortie : str, default="figures"
        Dossier de destination des figures si `save_fig=True`. Créé
        automatiquement s'il n'existe pas.

    Examples
    --------
    >>> from mathmorpho import MathMorphology, MorphologyStats
    >>> morpho = MathMorphology(volume)
    >>> apres = morpho.ouverture(rayon=1)
    >>> stats = MorphologyStats(volume, apres, save_fig=True)
    >>> resume = stats.resume()
    >>> stats.rapport_complet()
    """

    def __init__(
        self,
        image_avant: np.ndarray,
        image_apres: np.ndarray,
        save_fig: bool = False,
        dossier_sortie: str = "figures",
    ):
        image_avant = np.asarray(image_avant)
        image_apres = np.asarray(image_apres)

        if image_avant.shape != image_apres.shape:
            raise ValueError(
                "image_avant et image_apres doivent avoir la même forme "
                f"(reçu {image_avant.shape} et {image_apres.shape})."
            )

        self.image_avant = image_avant
        self.image_apres = image_apres
        self.save_fig = save_fig
        self.dossier_sortie = dossier_sortie

        if self.save_fig:
            os.makedirs(self.dossier_sortie, exist_ok=True)

    # ------------------------------------------------------------------
    # Utilitaires internes
    # ------------------------------------------------------------------
    def _est_binaire(self, image: np.ndarray) -> bool:
        return np.unique(image).size <= 2

    def _compter_composantes(self, image: np.ndarray) -> int:
        """Compte le nombre de composantes connexes (image binarisée > 0)."""
        masque_binaire = image > 0
        _, nb_composantes = _scipy_label(masque_binaire)
        return int(nb_composantes)

    def _sauvegarder_ou_afficher(self, fig, nom_fichier: str) -> None:
        """Sauvegarde la figure en .png si save_fig=True, sinon l'affiche."""
        if self.save_fig:
            chemin = os.path.join(self.dossier_sortie, nom_fichier)
            fig.savefig(chemin, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Figure sauvegardée : {chemin}")
        else:
            plt.show()

    # ------------------------------------------------------------------
    # Résumé statistique comparatif
    # ------------------------------------------------------------------
    def resume(self) -> dict:
        """
        Calcule un résumé statistique comparatif entre l'image avant et
        après transformation.

        Quand l'utiliser
        -----------------
        - Après toute transformation morphologique, pour vérifier
          objectivement son effet (combien de voxels ajoutés/supprimés,
          la région est-elle restée un seul bloc ou a-t-elle été
          fragmentée/reconnectée).
        - Pour documenter et justifier un choix de paramètres (rayon,
          forme) dans un pipeline de prétraitement.

        Returns
        -------
        dict
            Dictionnaire contenant :

            - ``volume_avant`` / ``volume_apres`` : nombre de voxels non nuls.
            - ``delta_volume`` : différence de volume (apres - avant).
            - ``delta_pourcentage`` : variation relative du volume, en %.
            - ``nb_voxels_ajoutes`` : voxels devenus non nuls après transformation.
            - ``nb_voxels_supprimes`` : voxels devenus nuls après transformation.
            - ``nb_composantes_avant`` / ``nb_composantes_apres`` : nombre
              de composantes connexes (utile pour vérifier fragmentation
              ou reconnexion des régions tumorales).

        Examples
        --------
        >>> resume = stats.resume()
        >>> print(resume["delta_pourcentage"])
        """
        masque_avant = self.image_avant > 0
        masque_apres = self.image_apres > 0

        volume_avant = int(masque_avant.sum())
        volume_apres = int(masque_apres.sum())
        delta_volume = volume_apres - volume_avant
        delta_pourcentage = (
            (delta_volume / volume_avant * 100.0) if volume_avant > 0 else float("nan")
        )

        voxels_ajoutes = int(np.logical_and(masque_apres, np.logical_not(masque_avant)).sum())
        voxels_supprimes = int(np.logical_and(masque_avant, np.logical_not(masque_apres)).sum())

        nb_composantes_avant = self._compter_composantes(self.image_avant)
        nb_composantes_apres = self._compter_composantes(self.image_apres)

        return {
            "volume_avant": volume_avant,
            "volume_apres": volume_apres,
            "delta_volume": delta_volume,
            "delta_pourcentage": delta_pourcentage,
            "nb_voxels_ajoutes": voxels_ajoutes,
            "nb_voxels_supprimes": voxels_supprimes,
            "nb_composantes_avant": nb_composantes_avant,
            "nb_composantes_apres": nb_composantes_apres,
        }

    def afficher_resume(self) -> None:
        """
        Affiche le résumé statistique de manière lisible dans la console.

        Examples
        --------
        >>> stats.afficher_resume()
        """
        r = self.resume()
        print("=" * 50)
        print("Résumé statistique comparatif (avant / après)")
        print("=" * 50)
        print(f"Volume avant           : {r['volume_avant']} voxels")
        print(f"Volume après            : {r['volume_apres']} voxels")
        print(f"Delta volume            : {r['delta_volume']} voxels "
              f"({r['delta_pourcentage']:.2f} %)")
        print(f"Voxels ajoutés          : {r['nb_voxels_ajoutes']}")
        print(f"Voxels supprimés        : {r['nb_voxels_supprimes']}")
        print(f"Composantes connexes avant : {r['nb_composantes_avant']}")
        print(f"Composantes connexes après : {r['nb_composantes_apres']}")
        print("=" * 50)

    # ------------------------------------------------------------------
    # Histogrammes
    # ------------------------------------------------------------------
    def histogramme_intensites(self, bins: int = 50):
        """
        Trace un histogramme comparatif des intensités (voxels non nuls)
        avant et après transformation.

        Parameters
        ----------
        bins : int, default=50
            Nombre de classes de l'histogramme.

        Returns
        -------
        matplotlib.figure.Figure
            La figure générée.

        Examples
        --------
        >>> stats.histogramme_intensites(bins=30)
        """
        valeurs_avant = self.image_avant[self.image_avant > 0].ravel()
        valeurs_apres = self.image_apres[self.image_apres > 0].ravel()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(valeurs_avant, bins=bins, alpha=0.5, label="Avant", color="steelblue")
        ax.hist(valeurs_apres, bins=bins, alpha=0.5, label="Après", color="darkorange")
        ax.set_xlabel("Intensité")
        ax.set_ylabel("Nombre de voxels")
        ax.set_title("Histogramme comparatif des intensités")
        ax.legend()
        fig.tight_layout()

        self._sauvegarder_ou_afficher(fig, "histogramme_intensites.png")
        return fig

    def histogramme_volume_comparatif(self):
        """
        Trace un diagramme en barres comparant le volume total (nombre de
        voxels non nuls) avant et après transformation.

        Returns
        -------
        matplotlib.figure.Figure
            La figure générée.

        Examples
        --------
        >>> stats.histogramme_volume_comparatif()
        """
        r = self.resume()

        fig, ax = plt.subplots(figsize=(5, 5))
        barres = ax.bar(
            ["Avant", "Après"],
            [r["volume_avant"], r["volume_apres"]],
            color=["steelblue", "darkorange"],
        )
        ax.set_ylabel("Volume (nombre de voxels)")
        ax.set_title("Comparaison du volume avant / après")
        ax.bar_label(barres, padding=3)
        fig.tight_layout()

        self._sauvegarder_ou_afficher(fig, "histogramme_volume_comparatif.png")
        return fig

    # ------------------------------------------------------------------
    # Visualisation de coupes 2D
    # ------------------------------------------------------------------
    def afficher_coupes(self, axe: str = "axial", indice: Optional[int] = None):
        """
        Affiche côte à côte une coupe 2D avant / après transformation,
        extraite d'un volume 3D.

        Parameters
        ----------
        axe : str, default='axial'
            'axial' (dernier axe), 'coronal' (2e axe), 'sagittal' (1er axe).
        indice : int, optional
            Indice de la coupe. Si None, la coupe centrale est utilisée.

        Returns
        -------
        matplotlib.figure.Figure
            La figure générée.

        Examples
        --------
        >>> stats.afficher_coupes(axe="axial")
        """
        if self.image_avant.ndim != 3:
            raise ValueError("afficher_coupes nécessite un volume 3D.")

        axes_map = {"sagittal": 0, "coronal": 1, "axial": 2}
        if axe not in axes_map:
            raise ValueError(f"axe doit être l'un de {list(axes_map.keys())}.")

        dim = axes_map[axe]
        if indice is None:
            indice = self.image_avant.shape[dim] // 2

        def extraire_coupe(volume, dim, indice):
            if dim == 0:
                return volume[indice, :, :]
            elif dim == 1:
                return volume[:, indice, :]
            else:
                return volume[:, :, indice]

        coupe_avant = extraire_coupe(self.image_avant, dim, indice)
        coupe_apres = extraire_coupe(self.image_apres, dim, indice)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(coupe_avant.T, cmap="gray", origin="lower")
        axes[0].set_title(f"Avant (coupe {axe}, indice {indice})")
        axes[0].axis("off")

        axes[1].imshow(coupe_apres.T, cmap="gray", origin="lower")
        axes[1].set_title(f"Après (coupe {axe}, indice {indice})")
        axes[1].axis("off")

        fig.tight_layout()

        self._sauvegarder_ou_afficher(fig, f"coupes_{axe}_{indice}.png")
        return fig

    # ------------------------------------------------------------------
    # Rapport complet
    # ------------------------------------------------------------------
    def rapport_complet(self, axe: str = "axial", indice: Optional[int] = None) -> dict:
        """
        Génère l'ensemble des statistiques et graphiques disponibles en un
        seul appel : résumé console, histogramme des intensités, histogramme
        de volume comparatif, et comparaison visuelle de coupes (si volume 3D).

        Parameters
        ----------
        axe : str, default='axial'
            Axe utilisé pour `afficher_coupes` si l'image est un volume 3D.
        indice : int, optional
            Indice de coupe utilisé pour `afficher_coupes`.

        Returns
        -------
        dict
            Le résumé statistique (identique à `resume()`).

        Examples
        --------
        >>> stats.rapport_complet()
        """
        self.afficher_resume()
        self.histogramme_intensites()
        self.histogramme_volume_comparatif()
        if self.image_avant.ndim == 3:
            self.afficher_coupes(axe=axe, indice=indice)
        return self.resume()
