"""Example 02 — Marker-driven watershed segmentation.

Markers are seeded from intensity quantiles (background vs. high-intensity
foreground).  A Gaussian-smoothed gradient magnitude serves as the
elevation map.
"""
import sys

import numpy as np

from segmed3d import (
    ImageIO,
    Preprocessor,
    WatershedSegmentation,
    Postprocessor,
    Visualizer,
)


def main(tumor_path: str):
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    vol = Preprocessor.normalize(Preprocessor.gaussian_smooth(vol, sigma=1.0))

    # Build markers from intensity quantiles
    markers = np.zeros_like(vol, dtype=np.int32)
    markers[vol < np.percentile(vol, 30)] = 1                # background
    markers[vol > np.percentile(vol, 90)] = 2                # foreground
    print(f"Background markers = {int((markers == 1).sum())}")
    print(f"Foreground markers = {int((markers == 2).sum())}")

    seg = WatershedSegmentation(vol, affine, hdr)
    seg.fit(markers=markers, gradient_method='gradient',
            connectivity=26, compactness=0.01)
    mask = seg.get_mask()

    mask = Postprocessor.clean_mask(mask, min_size=50)
    seg.save('mask_watershed.nii.gz')

    Visualizer.plot_3d_slices(vol, mask)
    print("Saved mask_watershed.nii.gz")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 02_watershed_nifti.py <tumor.nii.gz>")
        sys.exit(1)
    main(sys.argv[1])
