import nibabel as nib
import numpy as np
import pytest

from preprocessing import Preprocessing


@pytest.fixture
def nifti_path(tmp_path):
    data = np.zeros((8, 8, 4), dtype=np.float32)
    data[2:6, 2:6, 1:3] = np.linspace(1.0, 10.0, 32).reshape(4, 4, 2)
    path = tmp_path / "volume.nii.gz"
    nib.save(nib.Nifti1Image(data, np.diag([1.2, 1.3, 1.4, 1.0])), path)
    return path, data


def test_preprocessing_scaling_and_normalization(nifti_path):
    path, data = nifti_path
    pre = Preprocessing()

    minmax = pre.min_max_global_scaling(str(path)).get_fdata()
    robust = pre.robust_min_max_scaling(str(path)).get_fdata()
    zscore = pre.z_score_global_normalization(str(path)).get_fdata()
    robust_z = pre.robust_z_score_normalization(str(path)).get_fdata()
    mad = pre.median_mad_scaling(str(path)).get_fdata()

    foreground = data > 0
    assert np.allclose(minmax[~foreground], 0)
    assert np.isclose(minmax[foreground].min(), 0)
    assert np.isclose(minmax[foreground].max(), 1)
    assert np.isclose(zscore[foreground].mean(), 0, atol=1e-6)
    assert robust_z.shape == data.shape
    assert mad.shape == data.shape
    assert np.all((robust[foreground] >= 0) & (robust[foreground] <= 1))


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("clahe", {"kernel_size": (4, 4)}),
        ("histogram_equalization", {}),
        ("gamma_correction", {"gamma": 0.8}),
        ("gaussian_mixture_model", {"n_components": 2, "return_mode": "hard_labels"}),
    ],
)
def test_preprocessing_intensity_methods_return_nifti(nifti_path, method, kwargs):
    path, data = nifti_path
    result = getattr(Preprocessing(), method)(str(path), **kwargs)

    assert isinstance(result, nib.Nifti1Image)
    assert result.shape == data.shape
    assert np.all(np.isfinite(result.get_fdata()))


def test_preprocessing_rejects_empty_foreground(nifti_path):
    path, _ = nifti_path

    with pytest.raises(ValueError, match="foreground mask is empty"):
        Preprocessing().min_max_global_scaling(str(path), foreground_threshold=100)
