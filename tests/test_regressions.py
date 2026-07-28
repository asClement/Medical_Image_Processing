import nibabel as nib
import numpy as np

from denoise import AnisotropicDenoiser
from medio.nifti import load_nifti
from segmed3d import Postprocessor


def test_anisotropic_denoiser_accepts_numpy_array():
    volume = np.zeros((8, 8, 4), dtype=np.float32)
    volume[2:6, 2:6, 1:3] = 1.0

    result = AnisotropicDenoiser(n_iter=1).filter(volume)

    assert isinstance(result, np.ndarray)
    assert result.shape == volume.shape
    assert result.dtype == np.float32


def test_extract_largest_n_keeps_largest_components():
    mask = np.zeros((12, 12, 4), dtype=np.uint8)
    mask[1:3, 1:3, 1:3] = 1  # 8 voxels
    mask[6:10, 6:10, 1:3] = 1  # 32 voxels

    largest = Postprocessor.extract_largest_n(mask, n=1)
    both = Postprocessor.extract_largest_n(mask, n=2)

    assert largest.sum() == 32
    assert both.sum() == 40


def test_load_nifti_applies_mask(tmp_path):
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    image = np.ones((4, 4, 2), dtype=np.float32)
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[1:3, 1:3, :] = 1
    affine = np.eye(4)

    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(mask, affine), mask_path)

    loaded = load_nifti(image_path, mask_path)

    assert np.array_equal(loaded.data, mask)
    assert np.array_equal(loaded.mask, mask)
