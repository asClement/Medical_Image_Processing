"""Tests for RegionGrowingSegmentation."""
import numpy as np
import pytest

from segmed3d import RegionGrowingSegmentation


def test_range_mode(synthetic_volume):
    vol, gt = synthetic_volume
    # Seed inside lesion 1 (centre at (40, 45, 32) per conftest, but seed is (x,y,z) → (45, 40, 32))
    seg = RegionGrowingSegmentation(vol)
    seg.fit(seed_point=(45, 40, 32), tolerance=40, mode='range', connectivity=26)
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert mask.sum() > 0


def test_gradient_mode(synthetic_volume):
    vol, _ = synthetic_volume
    seg = RegionGrowingSegmentation(vol)
    seg.fit(seed_point=(45, 40, 32), tolerance=10, mode='gradient', connectivity=26)
    mask = seg.get_mask()
    assert mask.sum() > 0


def test_seed_value_accessor(synthetic_volume):
    vol, _ = synthetic_volume
    seg = RegionGrowingSegmentation(vol)
    seg.fit(seed_point=(45, 40, 32), tolerance=20)
    v = seg.get_seed_value()
    assert isinstance(v, float)


def test_out_of_bounds_seed(synthetic_volume):
    vol, _ = synthetic_volume
    seg = RegionGrowingSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(seed_point=(-1, 0, 0))


def test_invalid_mode(synthetic_volume):
    vol, _ = synthetic_volume
    seg = RegionGrowingSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(seed_point=(10, 10, 10), mode='invalid')
