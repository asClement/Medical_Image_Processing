import numpy as np

from mathmorpho import NiftiIO


def test_sauvegarde_et_chargement_nifti(tmp_path):
    volume = np.random.rand(10, 10, 10).astype(np.float32)
    affine = np.eye(4)
    chemin = tmp_path / "test.nii.gz"

    NiftiIO.sauvegarder_nifti(volume, affine, str(chemin))
    charge, affine_chargee, header = NiftiIO.charger_nifti(str(chemin))

    assert charge.shape == volume.shape
    assert np.allclose(affine, affine_chargee)
    assert header is not None
