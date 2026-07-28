# segmed3d

**Mathematical morphology & segmentation library for 3D medical images (NIfTI).**

`segmed3d` provides a unified, scikit-learn-style API for seven classical 3D
segmentation algorithms operating on NIfTI volumes (`.nii` / `.nii.gz`),
with a focus on tumor and brain MRI/CT imaging.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features

| # | Algorithm | Module | Highlights |
|---|-----------|--------|------------|
| 1 | **Otsu / Multi-Otsu** | `threshold` | Global, multi-class, slice-wise |
| 2 | **Watershed** | `watershed` | Gradient / distance / image, 6/18/26-conn |
| 3 | **Active Contours** | `active_contour` | Slice-wise snakes + 3D morphological GAC |
| 4 | **Level Set (Chan-Vese)** | `level_set` | 2D slice-wise CV + 3D morphological ACWE/GAC |
| 5 | **Region Growing** | `region_growing` | 26-connectivity, range & gradient modes |
| 6 | **K-Means / Fuzzy C-Means** | `clustering` | Intensity + spatial features |
| 7 | **Multi-Atlas** | `atlas` | SimpleITK registration + 4 fusion strategies |

Plus a `utils` subpackage with:

- **ImageIO** — NIfTI load/save
- **Preprocessor** — normalise, denoise (Gaussian/median/bilateral), N4 bias correction
- **Postprocessor** — mask cleanup, hole filling, morphology, connected-component analysis
- **Metrics** — Dice, IoU, sensitivity, specificity, precision, HD95, volume similarity
- **Visualizer** — orthogonal slices, overlay, 3D surface, histogram

---

## Installation

```bash
# From source (development)
git clone https://github.com/segmed3d/segmed3d.git
cd segmed3d
pip install -e .

# With the optional Fuzzy C-Means dependency
pip install -e ".[fuzzy]"

# With development tools
pip install -e ".[dev]"
```

### Dependencies

- Python ≥ 3.8
- `numpy`, `scipy`, `scikit-image`, `scikit-learn`
- `nibabel` (NIfTI I/O), `SimpleITK` (registration + N4)
- `matplotlib`, `tqdm`
- Optional: `scikit-fuzzy` (for Fuzzy C-Means)

---

## Quick start

```python
from segmed3d import (
    ImageIO, Preprocessor, ThresholdSegmentation,
    Postprocessor, Visualizer, Metrics,
)

# 1. Load
vol, affine, hdr = ImageIO.load_nifti("tumor.nii.gz")

# 2. Preprocess — N4 bias correction + min-max normalisation
vol = Preprocessor.bias_field_correction(vol)
vol = Preprocessor.normalize(vol)

# 3. Segment — one-liner
mask = ThresholdSegmentation(vol, affine, hdr)(method='otsu')

# 4. Clean up
mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)

# 5. Save & visualise
ImageIO.save_mask(mask, "mask.nii.gz", affine=affine, header=hdr)
Visualizer.plot_3d_slices(vol, mask)

# 6. Evaluate
gt, _, _ = ImageIO.load_mask("ground_truth.nii.gz")
print(Metrics.all_metrics(mask, gt, voxel_spacing=ImageIO.get_voxel_spacing(affine)))
```

---

## API pattern

Every segmentation class follows the same scikit-learn-style contract:

```python
class XxxSegmentation(BaseSegmentation):
    def __init__(self, volume, affine=None, header=None): ...
    def fit(self, **kwargs) -> 'XxxSegmentation': ...   # algorithm-specific
    def get_mask(self) -> np.ndarray: ...               # uint8 binary mask
    def save(self, path: str) -> str: ...               # NIfTI output
    def __call__(self, **kwargs) -> np.ndarray: ...     # fit + get_mask
```

So you can either chain calls explicitly:

```python
seg = ThresholdSegmentation(vol, affine, hdr)
seg.fit(method='otsu')
mask = seg.get_mask()
seg.save("mask.nii.gz")
```

…or use the one-liner shortcut:

```python
mask = ThresholdSegmentation(vol, affine, hdr)(method='otsu')
```

---

## Algorithm details

### 1. Threshold (Otsu)
- **Global Otsu** — optimal inter-class variance threshold over the full 3D histogram.
- **Multi-Otsu** — N-class extension; foreground = brightest class.
- **Slice-wise Otsu** — robust against intensity drift across slices.

### 2. Watershed
- Marker-driven, with three elevation maps: gradient magnitude (Sobel),
  distance transform, or inverted image.
- 6 / 18 / 26-connectivity, optional compactness term.

### 3. Active Contour
- `slice_wise`: classic parametric snakes (`skimage.segmentation.active_contour`)
  applied on each Z-slice, contours rasterised back to a 3D mask.
- `morphological_geodesic`: true 3D morphological GAC, evolving a level set
  guided by an edge indicator `g = 1 / (1 + |∇I|)`.

### 4. Level Set
- `chan_vese`: 2D slice-wise region-based Chan-Vese.
- `morphological_chan_vese`: 3D morphological ACWE (fast, no PDE).
- `morphological_geodesic`: 3D edge-based GAC.

### 5. Region Growing
- 26-connectivity BFS from a seed voxel.
- `range` mode: voxels within `[seed ± tol]`.
- `gradient` mode: per-edge intensity difference ≤ `tol` (robust to smooth ramps).

### 6. Clustering
- K-Means and Fuzzy C-Means on an intensity + spatial feature space.
- Foreground cluster = highest mean intensity.
- Fuzzy membership volume available via `get_membership_volume()`.

### 7. Multi-Atlas
- Optional registration (rigid / affine / Demons / SyN) via SimpleITK.
- Four label-fusion strategies:
  - **Majority voting** — hard vote.
  - **Weighted voting** — intensity-agreement-weighted soft vote.
  - **STAPLE** — Simultaneous Truth And Performance Level Estimation.
  - **JLF** — simplified Joint Label Fusion.

---

## Examples

Seven ready-to-run scripts live in `examples/`:

| File | Algorithm |
|------|-----------|
| `01_otsu_nifti.py` | Otsu |
| `02_watershed_nifti.py` | Watershed |
| `03_active_contour_nifti.py` | Active Contour |
| `04_level_set_nifti.py` | Level Set |
| `05_region_growing_nifti.py` | Region Growing |
| `06_clustering_nifti.py` | K-Means + FCM |
| `07_multi_atlas_brain.py` | Multi-Atlas |

Run any of them with:

```bash
python examples/01_otsu_nifti.py path/to/tumor.nii.gz [ground_truth.nii.gz]
```

---

## Testing

```bash
pytest -q
```

Tests use synthetic 3D volumes with embedded lesions — no external data
required.

---

## Project structure

```
segmed3d/
├── segmed3d/
│   ├── __init__.py              # Public API
│   ├── _base.py                 # BaseSegmentation(ABC)
│   ├── _version.py
│   ├── threshold.py             # ThresholdSegmentation
│   ├── watershed.py             # WatershedSegmentation
│   ├── active_contour.py        # ActiveContourSegmentation
│   ├── level_set.py             # LevelSetSegmentation
│   ├── region_growing.py        # RegionGrowingSegmentation
│   ├── clustering.py            # ClusteringSegmentation
│   ├── atlas.py                 # AtlasSegmentation
│   ├── py.typed
│   └── utils/
│       ├── __init__.py
│       ├── io.py                # ImageIO
│       ├── preprocessing.py     # Preprocessor
│       ├── postprocessing.py    # Postprocessor
│       ├── metrics.py           # Metrics
│       └── visualization.py     # Visualizer
├── tests/
│   ├── conftest.py
│   ├── test_threshold.py
│   ├── test_watershed.py
│   ├── test_active_contour.py
│   ├── test_level_set.py
│   ├── test_region_growing.py
│   ├── test_clustering.py
│   └── test_atlas.py
├── examples/
│   ├── 01_otsu_nifti.py
│   ├── 02_watershed_nifti.py
│   ├── 03_active_contour_nifti.py
│   ├── 04_level_set_nifti.py
│   ├── 05_region_growing_nifti.py
│   ├── 06_clustering_nifti.py
│   └── 07_multi_atlas_brain.py
├── pyproject.toml
├── README.md
├── CHANGELOG.md
└── LICENSE
```

---

## License

MIT — see [LICENSE](LICENSE).
