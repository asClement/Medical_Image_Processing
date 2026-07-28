"""Tests for WatershedSegmentation."""
import numpy as np
import pytest

from segmed3d import WatershedSegmentation


def test_watershed_gradient(synthetic_volume):
    vol, gt = synthetic_volume
    markers = np.zeros_like(vol, dtype=np.int32)
    markers[vol < vol.mean()] = 1
    markers[vol > vol.mean() + 2 * vol.std()] = 2
    seg = WatershedSegmentation(vol)
    seg.fit(markers=markers, gradient_method='gradient')
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_watershed_image(synthetic_volume):
    vol, _ = synthetic_volume
    markers = np.zeros_like(vol, dtype=np.int32)
    markers[vol < vol.mean()] = 1
    markers[vol > vol.mean() + 2 * vol.std()] = 2
    seg = WatershedSegmentation(vol)
    seg.fit(markers=markers, gradient_method='image')
    assert seg.get_mask().shape == vol.shape


def test_watershed_invalid_markers(synthetic_volume):
    vol, _ = synthetic_volume
    seg = WatershedSegmentation(vol)
    markers = np.ones_like(vol, dtype=np.int32)  # only one label
    with pytest.raises(ValueError):
        seg.fit(markers=markers)


def test_watershed_get_labels(synthetic_volume):
    vol, _ = synthetic_volume
    markers = np.zeros_like(vol, dtype=np.int32)
    markers[vol < vol.mean()] = 1
    markers[vol > vol.mean() + 2 * vol.std()] = 2
    seg = WatershedSegmentation(vol)
    seg.fit(markers=markers, gradient_method='gradient')
    labels = seg.get_labels()
    assert labels.shape == vol.shape
    assert labels.dtype == np.int32 or np.issubdtype(labels.dtype, np.integer)
