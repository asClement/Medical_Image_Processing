"""Tests for LevelSetSegmentation."""
import numpy as np

from segmed3d import LevelSetSegmentation


def test_morphological_chan_vese(synthetic_volume):
    vol, _ = synthetic_volume
    init = np.zeros_like(vol, dtype=np.uint8)
    init[30:50, 30:50, 20:40] = 1
    seg = LevelSetSegmentation(vol)
    seg.fit(init_mask=init, method='morphological_chan_vese', iterations=20)
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_chan_vese_slice_wise(synthetic_volume):
    vol, _ = synthetic_volume
    init = np.zeros_like(vol, dtype=np.uint8)
    init[30:50, 30:50, 20:40] = 1
    seg = LevelSetSegmentation(vol)
    seg.fit(init_mask=init, method='chan_vese', iterations=10)
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_unknown_method_raises(synthetic_volume):
    vol, _ = synthetic_volume
    init = np.zeros_like(vol, dtype=np.uint8)
    seg = LevelSetSegmentation(vol)
    try:
        seg.fit(init_mask=init, method='foo')
        assert False
    except ValueError:
        pass
