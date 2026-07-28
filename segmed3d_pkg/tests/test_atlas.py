"""Tests for AtlasSegmentation."""
import numpy as np
import pytest

from segmed3d import AtlasSegmentation


@pytest.fixture(scope="module")
def simple_atlases(synthetic_volume):
    """Build two synthetic atlases by translating the ground-truth mask."""
    vol, gt = synthetic_volume
    atlases_img = []
    atlases_lbl = []
    for shift in [(-2, 0, 0), (2, 0, 0), (0, -2, 0)]:
        img_s = np.roll(vol, shift, axis=(0, 1, 2))
        lbl_s = np.roll(gt, shift, axis=(0, 1, 2))
        atlases_img.append(img_s.astype(np.float32))
        atlases_lbl.append(lbl_s.astype(np.int16))
    return atlases_img, atlases_lbl


def test_majority_voting_no_registration(synthetic_volume, simple_atlases):
    vol, _ = synthetic_volume
    atlases_img, atlases_lbl = simple_atlases
    seg = AtlasSegmentation(vol)
    seg.fit(
        atlas_images=atlases_img,
        atlas_labels=atlases_lbl,
        method='majority_voting',
        register=False,
    )
    mask = seg.get_mask()
    assert mask.shape == vol.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_weighted_voting(synthetic_volume, simple_atlases):
    vol, _ = synthetic_volume
    atlases_img, atlases_lbl = simple_atlases
    seg = AtlasSegmentation(vol)
    seg.fit(
        atlas_images=atlases_img,
        atlas_labels=atlases_lbl,
        method='weighted_voting',
        register=False,
    )
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_staple(synthetic_volume, simple_atlases):
    vol, _ = synthetic_volume
    atlases_img, atlases_lbl = simple_atlases
    seg = AtlasSegmentation(vol)
    seg.fit(
        atlas_images=atlases_img,
        atlas_labels=atlases_lbl,
        method='STAPLE',
        register=False,
    )
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_jlf(synthetic_volume, simple_atlases):
    vol, _ = synthetic_volume
    atlases_img, atlases_lbl = simple_atlases
    seg = AtlasSegmentation(vol)
    seg.fit(
        atlas_images=atlases_img,
        atlas_labels=atlases_lbl,
        method='JLF',
        register=False,
    )
    mask = seg.get_mask()
    assert mask.shape == vol.shape


def test_invalid_method(synthetic_volume, simple_atlases):
    vol, _ = synthetic_volume
    atlases_img, atlases_lbl = simple_atlases
    seg = AtlasSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(atlas_images=atlases_img, atlas_labels=atlases_lbl,
                method='foo', register=False)


def test_empty_atlases(synthetic_volume):
    vol, _ = synthetic_volume
    seg = AtlasSegmentation(vol)
    with pytest.raises(ValueError):
        seg.fit(atlas_images=[], atlas_labels=[])
