import numpy as np

from mathmorpho import DistanceTransform


def test_transformee_distance_basique():
    volume = np.zeros((11, 11, 11), dtype=np.uint8)
    volume[3:8, 3:8, 3:8] = 1
    dt = DistanceTransform(volume)
    carte = dt.transformee_distance()
    assert carte.shape == volume.shape
    assert carte.max() > 0
    # le centre doit avoir la plus grande distance au bord
    assert carte[5, 5, 5] == carte.max()


def test_transformee_distance_avec_sampling():
    volume = np.zeros((11, 11, 11), dtype=np.uint8)
    volume[3:8, 3:8, 3:8] = 1
    dt = DistanceTransform(volume)
    carte_uniforme = dt.transformee_distance()
    carte_sampling = dt.transformee_distance(sampling=(1.0, 1.0, 2.0))
    assert not np.allclose(carte_uniforme, carte_sampling)
