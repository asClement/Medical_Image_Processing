"""
mathmorpho.io
=============

Chargement et sauvegarde de fichiers NIfTI (.nii / .nii.gz) en conservant
les métadonnées (matrice affine, header), indispensables pour recaler
correctement les volumes traités dans l'espace patient.
"""

from __future__ import annotations

import numpy as np
import nibabel as nib


class NiftiIO:
    """
    Classe utilitaire (méthodes statiques) pour charger et sauvegarder des
    fichiers NIfTI (.nii ou .nii.gz).

    Exemples
    --------
    >>> from mathmorpho import NiftiIO
    >>> volume, affine, header = NiftiIO.charger_nifti("segmentation.nii.gz")
    >>> NiftiIO.sauvegarder_nifti(volume, affine, "copie.nii.gz", header)
    """

    @staticmethod
    def charger_nifti(chemin: str):
        """
        Charge un fichier NIfTI (.nii ou .nii.gz).

        Parameters
        ----------
        chemin : str
            Chemin vers le fichier .nii ou .nii.gz.

        Returns
        -------
        array : np.ndarray
            Volume 3D (ou 4D) sous forme de tableau numpy.
        affine : np.ndarray
            Matrice affine (4x4) associée au volume.
        header : nibabel.Nifti1Header
            Header original du fichier NIfTI (utile pour la sauvegarde).

        Examples
        --------
        >>> volume, affine, header = NiftiIO.charger_nifti("irm.nii.gz")
        """
        img = nib.load(chemin)
        array = img.get_fdata()
        affine = img.affine
        header = img.header
        return array, affine, header

    @staticmethod
    def sauvegarder_nifti(array: np.ndarray, affine: np.ndarray, chemin: str, header=None) -> None:
        """
        Sauvegarde un tableau numpy en fichier NIfTI (.nii.gz recommandé).

        Parameters
        ----------
        array : np.ndarray
            Volume 3D à sauvegarder.
        affine : np.ndarray
            Matrice affine (4x4) à associer au volume (généralement celle
            de l'image d'origine, pour rester dans le même repère spatial).
        chemin : str
            Chemin de destination (ex : "resultat.nii.gz").
        header : nibabel.Nifti1Header, optionnel
            Header d'origine à réutiliser (recommandé pour préserver les
            métadonnées comme le voxel spacing).

        Examples
        --------
        >>> NiftiIO.sauvegarder_nifti(volume, affine, "resultat.nii.gz", header)
        """
        img = nib.Nifti1Image(array, affine=affine, header=header)
        nib.save(img, chemin)
