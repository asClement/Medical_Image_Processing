"""Example 07 — Multi-atlas segmentation (brain-style pipeline).

Two pre-aligned atlases (image + label) are fused to segment a target
volume.  All four fusion strategies are demonstrated.

If your atlases are NOT pre-aligned, set ``register=True`` and choose a
registration method ('demons', 'syn', 'affine', 'rigid').
"""
import sys

from segmed3d import (
    ImageIO,
    AtlasSegmentation,
    Postprocessor,
    Visualizer,
    Metrics,
)


def main(target_path: str, atlas_img_paths: list, atlas_lbl_paths: list):
    target, affine, hdr = ImageIO.load_nifti(target_path)

    atlas_imgs = [ImageIO.load_nifti(p)[0] for p in atlas_img_paths]
    atlas_lbls = [ImageIO.load_mask(p)[0] for p in atlas_lbl_paths]

    for method in ['majority_voting', 'weighted_voting', 'STAPLE', 'JLF']:
        print(f"\n=== Fusion: {method} ===")
        seg = AtlasSegmentation(target, affine, hdr)
        seg.fit(
            atlas_images=atlas_imgs,
            atlas_labels=atlas_lbls,
            method=method,
            register=False,  # set True if atlases are not pre-aligned
        )
        mask = seg.get_mask()
        mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)
        out_path = f'mask_atlas_{method}.nii.gz'
        seg.save(out_path)
        print(f"Saved {out_path}  ({int(mask.sum())} voxels)")

    Visualizer.plot_3d_slices(target, mask)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python 07_multi_atlas_brain.py <target.nii.gz> "
            "<atlas1_img.nii.gz> <atlas1_lbl.nii.gz> "
            "<atlas2_img.nii.gz> <atlas2_lbl.nii.gz> ..."
        )
        sys.exit(1)
    args = sys.argv[1:]
    target_path = args[0]
    rest = args[1:]
    if len(rest) % 2 != 0:
        print("Atlas paths must come in pairs: img lbl img lbl ...")
        sys.exit(1)
    img_paths = rest[0::2]
    lbl_paths = rest[1::2]
    main(target_path, img_paths, lbl_paths)
