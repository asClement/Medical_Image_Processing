"""Tests for ThresholdSegmentation."""
import numpy as np
import pytest

from segmed3d import ThresholdSegmentation
from segmed3d.utils import Metrics


def test_otsu_returns_binary(synthetic_volume):
    vol, gt = synthetic_volume
    seg = ThresholdSegmentation(vol)
    seg.fit(method='otsu')
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_otsu_threshold_value(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ThresholdSegmentation(vol)
    seg.fit(method='otsu')
    t = seg.get_threshold()
    assert isinstance(t, float)


def test_multi_otsu(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ThresholdSegmentation(vol)
    seg.fit(method='multi_otsu', n_classes=3)
    thresholds = seg.get_threshold()
    assert isinstance(thresholds, list)
    assert len(thresholds) == 2


def test_slice_wise_otsu(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ThresholdSegmentation(vol)
    seg.fit(method='otsu', slice_wise=True)
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_call_shortcut(synthetic_volume):
    vol, _ = synthetic_volume
    mask = ThresholdSegmentation(vol)(method='otsu')
    assert mask.shape == vol.shape


def test_dice_above_random(synthetic_volume):
    vol, gt = synthetic_volume
    mask = ThresholdSegmentation(vol)(method='otsu')
    d = Metrics.dice(mask, gt)
    assert d > 0.3  # not perfect, but well above random
