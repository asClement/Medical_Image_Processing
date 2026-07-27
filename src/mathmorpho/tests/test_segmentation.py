import numpy as np

from mathmorpho import WatershedSegmentation


def test_segmenter_objet_unique():
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[5:15, 5:15, 5:15] = 1
    ws = WatershedSegmentation(volume)
    labels = ws.segmenter(distance_min=3)
    assert labels.shape == volume.shape
    assert labels.max() >= 1


def test_segmenter_avec_marqueurs_fournis():
    volume = np.zeros((20, 20, 20), dtype=np.uint8)
    volume[5:15, 5:15, 5:15] = 1

    marqueurs = np.zeros_like(volume, dtype=np.int32)
    marqueurs[9, 9, 9] = 1

    ws = WatershedSegmentation(volume)
    labels = ws.segmenter(marqueurs=marqueurs)
    assert labels.max() == 1
