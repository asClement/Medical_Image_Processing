"""Example 03 — Active contour (morphological GAC).

Initialise the contour from a coarse Otsu mask, then refine with the 3D
morphological geodesic active contour.
"""
import sys

from segmed3d import (
    ImageIO,
    Preprocessor,
    ThresholdSegmentation,
    ActiveContourSegmentation,
    Postprocessor,
    Visualizer,
)


def main(tumor_path: str):
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    vol = Preprocessor.normalize(vol)

    # Coarse initialisation
    init = ThresholdSegmentation(vol, affine, hdr)(method='otsu')
    init = Postprocessor.largest_cc(init)
    print(f"Init mask voxels = {int(init.sum())}")

    # Refine with morphological GAC
    seg = ActiveContourSegmentation(vol, affine, hdr)
    seg.fit(
        init_mask=init,
        method='morphological_geodesic',
        iterations=50,
        smoothing=1,
        threshold='auto',
        balloon=0.0,
        sigma=1.0,
    )
    mask = seg.get_mask()
    mask = Postprocessor.fill_holes(mask)
    print(f"Refined mask voxels = {int(mask.sum())}")

    seg.save('mask_active_contour.nii.gz')
    Visualizer.plot_3d_slices(vol, mask)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 03_active_contour_nifti.py <tumor.nii.gz>")
        sys.exit(1)
    main(sys.argv[1])
