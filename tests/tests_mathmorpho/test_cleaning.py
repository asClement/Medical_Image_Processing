import numpy as np
import pytest

from mathmorpho import MorphologyCleaning


@pytest.fixture
def volume_avec_bruit():
    volume = np.zeros((30, 30, 30), dtype=np.uint8)
    volume[10:20, 10:20, 10:20] = 1  # objet principal (1000 voxels)
    volume[2, 2, 2] = 1               # artefact isole (1 voxel)
    volume[15, 15, 15] = 0            # petit trou
    return volume


def test_supprimer_petits_objets(volume_avec_bruit):
    clean = MorphologyCleaning(volume_avec_bruit)
    resultat = clean.supprimer_petits_objets(taille_min=10)
    assert resultat[2, 2, 2] == False
    assert resultat[12, 12, 12] == True


def test_supprimer_petits_trous(volume_avec_bruit):
    clean = MorphologyCleaning(volume_avec_bruit)
    resultat = clean.supprimer_petits_trous(taille_min=10)
    assert resultat[15, 15, 15] == True


def test_etiqueter_composantes(volume_avec_bruit):
    clean = MorphologyCleaning(volume_avec_bruit)
    labels, nb, props = clean.etiqueter_composantes()
    assert nb == 2
    volumes = sorted(p["volume"] for p in props)
    assert volumes[0] == 1        # artefact isole
    assert volumes[1] == 999      # objet principal (1000 - 1 trou)


def test_garder_plus_grande_composante(volume_avec_bruit):
    clean = MorphologyCleaning(volume_avec_bruit)
    resultat = clean.garder_plus_grande_composante()
    assert resultat[2, 2, 2] == False
    assert resultat[12, 12, 12] == True
    assert resultat.sum() == 999


def test_garder_plus_grande_composante_image_vide():
    vide = np.zeros((10, 10, 10), dtype=np.uint8)
    clean = MorphologyCleaning(vide)
    resultat = clean.garder_plus_grande_composante()
    assert resultat.sum() == 0
