"""Example 01 — Otsu thresholding on a NIfTI tumor volume.

Pipeline
--------
1. Load the NIfTI volume.
2. N4 bias-field correction + min-max normalisation.
3. Global Otsu thresholding.
4. Mask cleanup (hole filling + small-object removal).
5. Save the mask and evaluate against the ground truth (if available).
"""
import sys

from segmed3d import (
    ImageIO,
    Preprocessor,
    ThresholdSegmentation,
    Postprocessor,
    Visualizer,
    Metrics,
)


def main(tumor_path: str, gt_path: str = None):
    # 1. Load
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    print(f"Loaded volume: shape={vol.shape}, dtype={vol.dtype}, "
          f"range=[{vol.min():.1f}, {vol.max():.1f}]")

    # 2. Preprocess
    vol = Preprocessor.bias_field_correction(vol)
    vol = Preprocessor.normalize(vol)

    # 3. Segment
    seg = ThresholdSegmentation(vol, affine, hdr)
    seg.fit(method='otsu')
    mask = seg.get_mask()
    print(f"Otsu threshold = {seg.get_threshold():.4f}")
    print(f"Mask voxels    = {int(mask.sum())}")

    # 4. Postprocess
    mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)

    # 5. Save
    seg.save('mask_otsu.nii.gz')
    print("Saved mask_otsu.nii.gz")

    # 6. Visualise
    Visualizer.plot_3d_slices(vol, mask)

    # 7. Evaluate (optional)
    if gt_path is not None:
        gt, _, _ = ImageIO.load_mask(gt_path)
        spacing = ImageIO.get_voxel_spacing(affine)
        print(Metrics.all_metrics(mask, gt, voxel_spacing=spacing))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 01_otsu_nifti.py <tumor.nii.gz> [ground_truth.nii.gz]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
