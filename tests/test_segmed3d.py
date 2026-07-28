import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from segmed3d import (
    ActiveContourSegmentation,
    AtlasSegmentation,
    ClusteringSegmentation,
    ImageIO,
    LevelSetSegmentation,
    Metrics,
    Postprocessor,
    Preprocessor,
    RegionGrowingSegmentation,
    ThresholdSegmentation,
    Visualizer,
    WatershedSegmentation,
)


@pytest.fixture
def synthetic_volume():
    volume = np.zeros((12, 12, 6), dtype=np.float32)
    volume[3:9, 3:9, 1:5] = 10.0
    volume[5:7, 5:7, 2:4] = 20.0
    return volume


def assert_binary_mask(mask, shape):
    assert mask.shape == shape
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_base_segmentation_threshold_and_save(synthetic_volume, tmp_path):
    seg = ThresholdSegmentation(synthetic_volume)
    mask = seg(method="otsu")
    output = seg.save(str(tmp_path / "mask.nii.gz"))

    assert_binary_mask(mask, synthetic_volume.shape)
    assert seg.get_threshold() is not None
    loaded, _, _ = ImageIO.load_mask(output)
    assert np.array_equal(loaded, mask)


def test_watershed_segmentation(synthetic_volume):
    markers = np.zeros_like(synthetic_volume, dtype=np.int32)
    markers[1, 1, 1] = 1
    markers[10, 10, 4] = 2

    seg = WatershedSegmentation(synthetic_volume).fit(markers, gradient_method="image")

    assert_binary_mask(seg.get_mask(), synthetic_volume.shape)
    assert seg.get_labels().shape == synthetic_volume.shape


def test_region_growing_and_clustering(synthetic_volume):
    region = RegionGrowingSegmentation(synthetic_volume).fit(
        seed_point=(4, 4, 2), tolerance=0.1, connectivity=6
    )
    clustering = ClusteringSegmentation(synthetic_volume).fit(
        method="kmeans", n_clusters=3, spatial_weight=0.0, random_state=0, max_iter=30
    )

    assert_binary_mask(region.get_mask(), synthetic_volume.shape)
    assert_binary_mask(clustering.get_mask(), synthetic_volume.shape)
    assert clustering.get_cluster_volume().shape == synthetic_volume.shape
    assert clustering.get_membership_volume() is None
    assert clustering.get_foreground_label() in {0, 1, 2}


@pytest.mark.parametrize(
    ("segmenter", "kwargs"),
    [
        (ActiveContourSegmentation, {"method": "morphological_geodesic", "iterations": 1}),
        (LevelSetSegmentation, {"method": "morphological_chan_vese", "iterations": 1}),
    ],
)
def test_contour_segmenters_accept_initial_mask(synthetic_volume, segmenter, kwargs):
    init = np.zeros_like(synthetic_volume, dtype=np.uint8)
    init[4:8, 4:8, 2:4] = 1

    result = segmenter(synthetic_volume).fit(init_mask=init, **kwargs).get_mask()

    assert_binary_mask(result, synthetic_volume.shape)


def test_atlas_fusion_without_registration(synthetic_volume):
    label = (synthetic_volume > 0).astype(np.uint8)
    seg = AtlasSegmentation(synthetic_volume).fit(
        atlas_images=[synthetic_volume, synthetic_volume],
        atlas_labels=[label, label],
        method="majority_voting",
        register=False,
    )

    assert_binary_mask(seg.get_mask(), synthetic_volume.shape)
    warped_images, warped_labels = seg.get_warped_atlases()
    assert len(warped_images) == len(warped_labels) == 2


def test_segmed3d_utilities_and_visualization(synthetic_volume):
    mask = (synthetic_volume > 0).astype(np.uint8)
    normalized = Preprocessor.normalize(synthetic_volume)
    cleaned = Postprocessor.clean_mask(mask, min_size=10, keep_largest=True)
    scores = Metrics.all_metrics(cleaned, mask)

    assert normalized.min() == 0
    assert normalized.max() == 1
    assert_binary_mask(cleaned, mask.shape)
    assert scores["dice"] == 1.0

    figures = [
        Visualizer.plot_3d_slices(synthetic_volume, mask),
        Visualizer.plot_overlay(synthetic_volume, mask),
        Visualizer.plot_3d_surface(mask),
        Visualizer.plot_histogram(synthetic_volume, mask=mask),
    ]
    assert all(figure.axes for figure in figures)
    for figure in figures:
        figure.clf()
