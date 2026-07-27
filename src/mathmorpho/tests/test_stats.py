import numpy as np
import pytest

from mathmorpho import MathMorphology, MorphologyStats


@pytest.fixture
def volumes_avant_apres():
    avant = np.zeros((20, 20, 20), dtype=np.uint8)
    avant[5:15, 5:15, 5:15] = 1
    avant[1, 1, 1] = 1  # artefact isole

    morpho = MathMorphology(avant)
    apres = morpho.ouverture(forme="ball", rayon=1)
    return avant, apres


def test_resume_contient_les_bonnes_cles(volumes_avant_apres):
    avant, apres = volumes_avant_apres
    stats = MorphologyStats(avant, apres)
    r = stats.resume()

    cles_attendues = {
        "volume_avant", "volume_apres", "delta_volume", "delta_pourcentage",
        "nb_voxels_ajoutes", "nb_voxels_supprimes",
        "nb_composantes_avant", "nb_composantes_apres",
    }
    assert cles_attendues.issubset(r.keys())


def test_ouverture_reduit_le_volume(volumes_avant_apres):
    avant, apres = volumes_avant_apres
    stats = MorphologyStats(avant, apres)
    r = stats.resume()
    assert r["volume_apres"] <= r["volume_avant"]


def test_composantes_connexes_diminuent_apres_ouverture(volumes_avant_apres):
    avant, apres = volumes_avant_apres
    stats = MorphologyStats(avant, apres)
    r = stats.resume()
    # l'artefact isole doit disparaitre -> moins de composantes apres
    assert r["nb_composantes_apres"] <= r["nb_composantes_avant"]


def test_shapes_differentes_leve_erreur():
    a = np.zeros((10, 10, 10))
    b = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        MorphologyStats(a, b)


def test_save_fig_cree_fichiers(volumes_avant_apres, tmp_path):
    avant, apres = volumes_avant_apres
    dossier = tmp_path / "figures"
    stats = MorphologyStats(avant, apres, save_fig=True, dossier_sortie=str(dossier))

    stats.histogramme_intensites()
    stats.histogramme_volume_comparatif()
    stats.afficher_coupes()

    fichiers = list(dossier.glob("*.png"))
    assert len(fichiers) == 3
