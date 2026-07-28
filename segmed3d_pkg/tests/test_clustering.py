"""Tests for ClusteringSegmentation."""
import numpy as np
import pytest

from segmed3d import ClusteringSegmentation


def test_kmeans(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ClusteringSegmentation(vol)
    seg.fit(method='kmeans', n_clusters=3, spatial_weight=0.3)
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert set(np.unique(mask)).issubset({0, 1})
    labels = seg.get_cluster_volume()
    assert labels.shape == vol.shape
    assert seg.get_membership_volume() is None


def test_kmeans_intensity_only(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ClusteringSegmentation(vol)
    seg.fit(method='kmeans', n_clusters=2, spatial_weight=0.0)
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_invalid_n_clusters(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ClusteringSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(method='kmeans', n_clusters=1)


def test_invalid_spatial_weight(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ClusteringSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(method='kmeans', n_clusters=2, spatial_weight=1.5)


def test_foreground_label(synthetic_volume):
    vol, _ = synthetic_volume
    seg = ClusteringSegmentation(vol)
    seg.fit(method='kmeans', n_clusters=3)
    lbl = seg.get_foreground_label()
    assert isinstance(lbl, int)
    assert lbl >= 0
