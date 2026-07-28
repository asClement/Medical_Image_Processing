"""Example 05 — Region growing from a seed point.

The seed is given in voxel coordinates.  Tolerance is set to ~20% of the
intensity dynamic range.  The 'gradient' mode is more robust to slow
intensity drift inside the lesion.
"""
import sys

from segmed3d import (
    ImageIO,
    Preprocessor,
    RegionGrowingSegmentation,
    Postprocessor,
    Visualizer,
)


def main(tumor_path: str, seed: tuple = (45, 40, 32)):
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    vol = Preprocessor.normalize(Preprocessor.gaussian_smooth(vol, sigma=0.5))

    rng = float(vol.max() - vol.min())
    tolerance = 0.2 * rng

    seg = RegionGrowingSegmentation(vol, affine, hdr)
    seg.fit(
        seed_point=seed,
        tolerance=tolerance,
        connectivity=26,
        mode='gradient',
    )
    mask = seg.get_mask()
    print(f"Seed value   = {seg.get_seed_value():.4f}")
    print(f"Tolerance    = {seg.get_seed_point()}")
    print(f"Mask voxels  = {int(mask.sum())}")

    mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)
    seg.save('mask_region_growing.nii.gz')
    Visualizer.plot_3d_slices(vol, mask)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05_region_growing_nifti.py <tumor.nii.gz> [x y z]")
        sys.exit(1)
    seed = (45, 40, 32)
    if len(sys.argv) >= 5:
        seed = tuple(int(v) for v in sys.argv[2:5])
    main(sys.argv[1], seed)
