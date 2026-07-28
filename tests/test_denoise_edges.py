import nibabel as nib
import numpy as np

from denoise import AnisotropicDenoiser, GaussianDenoiser, NLMRicianDenoiser
from edges import CannyEdgeDetector, SobelEdgeDetector
from medio import MedicalImage3D


def volume_and_mask():
    volume = np.zeros((16, 16, 4), dtype=np.float32)
    volume[4:12, 4:12, 1:3] = 1.0
    mask = np.zeros_like(volume, dtype=np.uint8)
    mask[4:12, 4:12, 1:3] = 1
    return volume, mask


def test_denoisers_support_numpy_and_medical_image():
    volume, mask = volume_and_mask()
    header = nib.Nifti1Image(volume, np.eye(4)).header
    image = MedicalImage3D(volume, np.eye(4), header, mask=mask)

    for denoiser in (
        GaussianDenoiser(sigma=0.5),
        AnisotropicDenoiser(n_iter=1),
        NLMRicianDenoiser(sigma=0.1),
    ):
        array_result = denoiser.filter(volume, mask=mask)
        image_result = denoiser.filter(image)
        assert array_result.shape == volume.shape
        assert image_result.data.shape == volume.shape
        assert image_result.data.dtype == np.float32


def test_edge_detectors_return_expected_shapes_and_masking():
    volume, mask = volume_and_mask()

    canny = CannyEdgeDetector(sigma=0.5).detect(volume)
    sobel = SobelEdgeDetector(apply_mask=True).detect(volume, mask)

    assert canny.shape == volume.shape
    assert sobel.shape == volume.shape
    assert set(np.unique(canny)).issubset({0, 1})
    assert np.all(sobel[mask == 0] == 0)
    assert np.any(canny)
    assert np.any(sobel)
