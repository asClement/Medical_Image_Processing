"""Tests for ActiveContourSegmentation."""
import numpy as np

from segmed3d import ActiveContourSegmentation, ThresholdSegmentation


def test_morphological_gac(synthetic_volume):
    vol, gt = synthetic_volume
    init = ThresholdSegmentation(vol)(method='otsu')
    seg = ActiveContourSegmentation(vol)
    seg.fit(init_mask=init, method='morphological_geodesic', iterations=10)
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_slice_wise(synthetic_volume):
    vol, _ = synthetic_volume
    init = ThresholdSegmentation(vol)(method='otsu')
    seg = ActiveContourSegmentation(vol)
    seg.fit(init_mask=init, method='slice_wise', max_iter=100)
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_shape_mismatch_raises(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ActiveContourSegmentation(vol)
    init = np.zeros((10, 10, 10), dtype=np.uint8)
    try:
        seg.fit(init_mask=init)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
