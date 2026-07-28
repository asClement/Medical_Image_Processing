# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-28

### Added
- Initial public release of `segmed3d`.
- Seven segmentation algorithms with a unified `fit()` / `get_mask()` /
  `__call__()` API:
  - `ThresholdSegmentation` (Otsu, multi-Otsu, slice-wise Otsu)
  - `WatershedSegmentation` (gradient / distance / image, 6/18/26-conn)
  - `ActiveContourSegmentation` (slice-wise snakes + 3D morphological GAC)
  - `LevelSetSegmentation` (Chan-Vese slice-wise + 3D morphological ACWE/GAC)
  - `RegionGrowingSegmentation` (26-conn, range & gradient modes)
  - `ClusteringSegmentation` (K-Means + Fuzzy C-Means)
  - `AtlasSegmentation` (SimpleITK registration + 4 fusion strategies:
    majority voting, weighted voting, STAPLE, JLF)
- `utils` subpackage:
  - `ImageIO` — NIfTI load/save, voxel spacing helper
  - `Preprocessor` — normalise, clip, rescale, Gaussian/median/bilateral
    denoising, N4 bias-field correction (SimpleITK)
  - `Postprocessor` — fill holes, remove small objects, largest-CC, morphological
    open/close/erode/dilate, all-in-one `clean_mask` pipeline
  - `Metrics` — Dice, IoU, sensitivity, specificity, precision, HD95
    (with medpy fallback), volume similarity, `all_metrics`
  - `Visualizer` — orthogonal slices, overlay, 3D surface, histogram
- `BaseSegmentation` ABC with input validation, NIfTI save, clone, repr.
- Seven runnable example scripts in `examples/`.
- Test-suite using synthetic volumes (no external data needed).
- `pyproject.toml` with optional `[fuzzy]` and `[dev]` extras.
- `py.typed` marker for PEP 561 type-checking support.

### Notes
- `scikit-fuzzy` is an **optional** dependency (FCM only) because the upstream
  project is poorly maintained.
- `active_contour` 3D is handled via a dual approach (slice-wise snakes + 3D
  morphological GAC) since `skimage.segmentation.active_contour` is 2D-only.
- `chan_vese` 3D uses the morphological ACWE (`morphological_chan_vese`),
  which is true 3D and PDE-free.

[Unreleased]: https://github.com/segmed3d/segmed3d/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/segmed3d/segmed3d/releases/tag/v0.1.0
