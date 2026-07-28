"""Example 06 — K-Means and Fuzzy C-Means clustering.

Compare two clustering strategies on the same volume.  The foreground
cluster is automatically chosen as the one with the highest mean intensity.
"""
import sys

from segmed3d import (
    ImageIO,
    Preprocessor,
    ClusteringSegmentation,
    Postprocessor,
    Visualizer,
)


def run(method: str, vol, affine, hdr):
    seg = ClusteringSegmentation(vol, affine, hdr)
    seg.fit(
        method=method,
        n_clusters=3,
        spatial_weight=0.3,
        random_state=42,
    )
    mask = seg.get_mask()
    mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)
    print(f"[{method}] foreground label = {seg.get_foreground_label()}")
    print(f"[{method}] mask voxels      = {int(mask.sum())}")
    return seg, mask


def main(tumor_path: str):
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    vol = Preprocessor.normalize(vol)

    print("\n=== K-Means ===")
    _, mask_km = run('kmeans', vol, affine, hdr)

    print("\n=== Fuzzy C-Means ===")
    try:
        seg_fcm, mask_fcm = run('fcm', vol, affine, hdr)
    except ImportError as e:
        print(f"FCM not available: {e}")
        return

    Visualizer.plot_3d_slices(vol, mask_km)
    Visualizer.plot_3d_slices(vol, mask_fcm)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 06_clustering_nifti.py <tumor.nii.gz>")
        sys.exit(1)
    main(sys.argv[1])
