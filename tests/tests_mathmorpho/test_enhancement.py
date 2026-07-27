import numpy as np
import pytest

from mathmorpho import MorphologyEnhancement


@pytest.fixture
def image_avec_petite_structure():
    img = np.zeros((30, 30, 30), dtype=np.float64)
    img[:, :, :] = 0.2
    img[13:17, 13:17, 13:17] = 0.9  # petite structure claire
    return img


def test_top_hat_blanc_extrait_structure_claire(image_avec_petite_structure):
    enh = MorphologyEnhancement(image_avec_petite_structure)
    resultat = enh.top_hat_blanc(forme="ball", rayon=2)
    assert resultat.shape == image_avec_petite_structure.shape
    assert resultat.max() > 0


def test_top_hat_noir_shape(image_avec_petite_structure):
    enh = MorphologyEnhancement(image_avec_petite_structure)
    resultat = enh.top_hat_noir(forme="ball", rayon=2)
    assert resultat.shape == image_avec_petite_structure.shape


def test_set_image(image_avec_petite_structure):
    enh = MorphologyEnhancement(image_avec_petite_structure)
    nouveau = np.ones((5, 5, 5))
    enh.set_image(nouveau)
    assert enh.image.shape == (5, 5, 5)


def test_forme_invalide_leve_erreur(image_avec_petite_structure):
    enh = MorphologyEnhancement(image_avec_petite_structure)
    with pytest.raises(ValueError):
        enh.top_hat_blanc(forme="triangle")
