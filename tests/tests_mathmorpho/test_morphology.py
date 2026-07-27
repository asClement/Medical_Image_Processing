import numpy as np
import pytest

from mathmorpho import MathMorphology


@pytest.fixture
def volume_binaire():
    volume = np.zeros((30, 30, 30), dtype=np.uint8)
    volume[10:20, 10:20, 10:20] = 1
    volume[2, 2, 2] = 1          # artefact isole
    volume[15, 15, 15] = 0       # trou
    return volume


def test_dilatation_augmente_le_volume(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.dilatation(iterations=1, forme="ball", rayon=1)
    assert resultat.sum() > volume_binaire.sum()


def test_erosion_diminue_le_volume(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.erosion(iterations=1, forme="ball", rayon=1)
    assert resultat.sum() < volume_binaire.sum()


def test_ouverture_supprime_artefact_isole(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.ouverture(forme="ball", rayon=1)
    assert resultat[2, 2, 2] == 0


def test_fermeture_comble_le_trou(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.fermeture(forme="ball", rayon=1)
    assert resultat[15, 15, 15] == 1


def test_reconstruction_dilatation(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    marqueur = morpho.erosion(rayon=1)
    resultat = morpho.reconstruction(marqueur, masque=volume_binaire, methode="dilatation")
    assert (resultat > 0).sum() >= marqueur.sum()


def test_erosion_geodesique(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    marqueur = morpho.dilatation(rayon=1)
    resultat = morpho.erosion_geodesique(marqueur, masque=marqueur)
    assert resultat.shape == volume_binaire.shape


def test_forme_cube(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.dilatation(forme="cube", rayon=1)
    assert resultat.sum() > volume_binaire.sum()


def test_forme_invalide_leve_erreur(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    with pytest.raises(ValueError):
        morpho.dilatation(forme="triangle")


def test_set_image(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    nouveau = np.ones((5, 5, 5))
    morpho.set_image(nouveau)
    assert morpho.image.shape == (5, 5, 5)


def test_gradient_morphologique_detecte_contours(volume_binaire):
    morpho = MathMorphology(volume_binaire)
    resultat = morpho.gradient_morphologique(forme="ball", rayon=1)
    assert resultat.shape == volume_binaire.shape
    assert resultat.max() > 0
    # loin de tout objet, le gradient doit etre nul
    assert resultat[0, 0, 0] == 0


def test_image_grayscale():
    gray = np.random.rand(20, 20, 20).astype(np.float32)
    morpho = MathMorphology(gray)
    assert morpho._est_binaire(gray) is False
    resultat = morpho.dilatation(forme="ball", rayon=1)
    assert resultat.shape == gray.shape
