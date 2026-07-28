from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np


@dataclass
class MedicalImage3D:
    data: np.ndarray
    affine: np.ndarray
    header: nib.Nifti2Header
    mask: np.ndarray | None = None
    mask_header: nib.Nifti2Header | None = None

    @property
    def spacing(self) -> Tuple[float, float, float]:
        zooms = self.header.get_zooms()
        return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def load_nifti(file_path: str | Path, mask_path: str | Path | None = None) -> MedicalImage3D:
    img = nib.load(str(file_path))
    data = img.get_fdata(dtype=np.float32)

    mask_data = None
    mask_header = None

    if mask_path:
        mask_img = nib.load(str(mask_path))
        mask_data = mask_img.get_fdata(dtype=np.float32)
        mask_data = np.round(mask_data).astype(np.int16)
        mask_header = mask_img.header
        if mask_data.shape != data.shape:
            raise ValueError("Le masque doit avoir la même forme que l'image")
        data = np.where(mask_data > 0, data, 0).astype(np.float32)

    return MedicalImage3D(
        data=data,
        affine=img.affine,
        header=img.header,
        mask=mask_data,
        mask_header=mask_header,
    )


def save_nifti(
    image: MedicalImage3D,
    outp_path: str | Path,
    save_mask: bool = False,
    mask_path: str | Path | None = None,
) -> None:
    new_img = nib.Nifti2Image(image.data, image.affine, header=image.header)
    nib.save(new_img, str(outp_path))
    if save_mask:
        if image.mask is None:
            raise ValueError("save_mask=True nécessite une image.mask non vide")
        if mask_path is None:
            raise ValueError("mask_path est requis lorsque save_mask=True")
        mask_img = nib.Nifti2Image(image.mask, image.affine, header=image.mask_header)
        nib.save(mask_img, str(mask_path))
