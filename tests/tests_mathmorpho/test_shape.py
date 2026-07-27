import numpy as np

from mathmorpho import MorphologyShape


def test_enveloppe_convexe_contient_objet():
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[5:15, 5:15, 5:15] = 1  # deja convexe (cube)
    forme = MorphologyShape(volume)
    enveloppe = forme.enveloppe_convexe()
    assert enveloppe.sum() >= volume.sum()


def test_indice_convexite_objet_convexe_proche_de_1():
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[5:15, 5:15, 5:15] = 1  # cube = deja convexe
    forme = MorphologyShape(volume)
    indice = forme.indice_convexite()
    assert 0.95 <= indice <= 1.0


def test_indice_convexite_objet_irregulier_plus_faible():
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    # forme en L, non convexe
    volume[5:15, 5:8, 5:8] = 1
    volume[5:8, 5:15, 5:8] = 1
    forme = MorphologyShape(volume)
    indice = forme.indice_convexite()
    assert indice < 0.95


def test_indice_convexite_image_vide():
    vide = np.zeros((10, 10, 10), dtype=np.uint8)
    forme = MorphologyShape(vide)
    indice = forme.indice_convexite()
    assert np.isnan(indice)
