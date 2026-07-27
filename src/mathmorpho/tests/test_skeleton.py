import numpy as np
import pytest

from mathmorpho import MorphologySkeleton


@pytest.fixture
def forme_allongee_2d():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 9:11] = 1
    return img


def test_squelettiser_reduit_le_volume(forme_allongee_2d):
    skel = MorphologySkeleton(forme_allongee_2d)
    resultat = skel.squelettiser()
    assert resultat.sum() > 0
    assert resultat.sum() < forme_allongee_2d.sum()


def test_squelettiser_3d():
    volume = np.zeros((15, 15, 15), dtype=np.uint8)
    volume[5:10, 6:9, 6:9] = 1
    skel = MorphologySkeleton(volume)
    resultat = skel.squelettiser()
    assert resultat.shape == volume.shape


def test_axe_median_2d(forme_allongee_2d):
    skel = MorphologySkeleton(forme_allongee_2d)
    axe, distance = skel.axe_median()
    assert axe.shape == forme_allongee_2d.shape
    assert distance.shape == forme_allongee_2d.shape


def test_axe_median_leve_erreur_en_3d():
    volume = np.zeros((10, 10, 10), dtype=np.uint8)
    skel = MorphologySkeleton(volume)
    with pytest.raises(ValueError):
        skel.axe_median()
