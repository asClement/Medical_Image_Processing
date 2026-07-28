import nibabel as nib
import numpy as np

from medio import MedicalImage3D, load_nifti, save_nifti
from segmed3d import ImageIO


def test_medio_round_trip_and_spacing(tmp_path):
    data = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    source = tmp_path / "source.nii.gz"
    output = tmp_path / "output.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), source)

    image = load_nifti(source)
    save_nifti(image, output)
    loaded = load_nifti(output)

    assert isinstance(image, MedicalImage3D)
    assert loaded.data.shape == data.shape
    assert np.allclose(loaded.data, data)
    assert loaded.spacing == (2.0, 3.0, 4.0)


def test_medio_mask_is_applied_and_saved(tmp_path):
    data = np.ones((4, 4, 2), dtype=np.float32)
    mask = np.zeros_like(data, dtype=np.uint8)
    mask[1:3, 1:3, :] = 1
    image_path = tmp_path / "image.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    output_mask = tmp_path / "output-mask.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)

    image = load_nifti(image_path, mask_path)
    save_nifti(image, tmp_path / "masked.nii.gz", save_mask=True, mask_path=output_mask)
    saved_mask, _, _ = ImageIO.load_mask(str(output_mask))

    assert np.array_equal(image.data, mask)
    assert np.array_equal(saved_mask, mask)


def test_imageio_round_trip_and_spacing(tmp_path):
    data = np.zeros((4, 5, 6), dtype=np.float32)
    affine = np.diag([1.5, 2.0, 2.5, 1.0])
    path = tmp_path / "image.nii.gz"

    saved = ImageIO.save_nifti(data, str(path), affine=affine)
    loaded, loaded_affine, _ = ImageIO.load_nifti(saved)

    assert np.array_equal(loaded, data)
    assert np.allclose(loaded_affine, affine)
    assert ImageIO.get_voxel_spacing(affine) == (1.5, 2.0, 2.5)
