# src/denoise/io/nifti.py
"""
Pour charger et sauvegarder des images NIFTI.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


# class de données médicales 3D
@dataclass
class MedicalImage3D:
    """Structure contenant les données 3D et leurs métadonnées spatiales."""

    data: np.ndarray  # Volume 3D (X, Y, Z)
    affine: np.ndarray  # Transformation affine
    header: nib.Nifti2Header  # En-tête NIFTI
    mask: np.ndarray | None = None
    mask_header: nib.Nifti2Header | None = None

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Retourne la taille des voxels"""
        zooms = self.header.get_zooms()
        return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


# Fonctions pour charger des images NIFTI
def load_nifti(
    file_path: str | Path, mask_path: str | Path | None = None
) -> MedicalImage3D:
    """Charge une image NIFTI depuis le chemin spécifié."""
    img = nib.load(str(file_path))
    data = img.get_fdata(dtype=np.float32)  # ty:ignore[unresolved-attribute]

    mask_data = None
    mask_header = None

    # Chargement du masque si spécifié
    if mask_path:
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata(dtype=np.float32)  # ty:ignore[unresolved-attribute]

        # binarisation du masque
        mask_data = np.round(mask_data).astype(np.int16)
        mask_header = mask_img.header

        # vérification de la forme du masque
        if mask_data.shape != data.shape:
            raise ValueError("Le masque doit avoir la même forme que l'image")

    return MedicalImage3D(
        data=data,
        affine=img.affine,  # ty:ignore[unresolved-attribute]
        header=img.header,  # ty:ignore[invalid-argument-type]
        mask=mask_data,
        mask_header=mask_header,  # ty:ignore[invalid-argument-type]
    )


# fonction pour sauvegarder une image NIFTI
def save_nifti(
    image: MedicalImage3D,
    outp_path: str | Path,
    save_mask: bool = False,
    mask_path: str | Path | None = None,
) -> None:
    """Sauvegarde le volume filtré en conservant la métadonnées originales."""
    new_img = nib.Nifti2Image(image.data, image.affine, header=image.header)
    nib.save(new_img, str(outp_path))

    # sauvegarde du masque si spécifié
    if save_mask:
        if image.mask is not None:
            mask_img = nib.Nifti2Image(image.mask, image.affine, header=image.mask_header)
            nib.save(mask_img, str(mask_path))
