"""Example 04 — Level-set segmentation (morphological Chan-Vese).

Initialise the level set with a small cube inside the suspected lesion
region and let the morphological ACWE evolve over 100 iterations.
"""
import sys

import numpy as np

from segmed3d import (
    ImageIO,
    Preprocessor,
    LevelSetSegmentation,
    Postprocessor,
    Visualizer,
)


def main(tumor_path: str, seed: tuple = (45, 40, 32)):
    vol, affine, hdr = ImageIO.load_nifti(tumor_path)
    vol = Preprocessor.normalize(vol)

    # Initialise level set as a small cube around the seed
    init = np.zeros_like(vol, dtype=np.uint8)
    sx, sy, sz = seed
    init[sx - 10:sx + 10, sy - 10:sy + 10, sz - 5:sz + 5] = 1

    seg = LevelSetSegmentation(vol, affine, hdr)
    seg.fit(
        init_mask=init,
        method='morphological_chan_vese',
        iterations=100,
        smoothing=1,
        lambda1=1.0,
        lambda2=1.0,
    )
    mask = seg.get_mask()
    mask = Postprocessor.clean_mask(mask, min_size=50, fill_holes=True)

    seg.save('mask_level_set.nii.gz')
    print(f"Mask voxels = {int(mask.sum())}")
    Visualizer.plot_3d_slices(vol, mask)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 04_level_set_nifti.py <tumor.nii.gz> [x y z]")
        sys.exit(1)
    seed = (45, 40, 32)
    if len(sys.argv) >= 5:
        seed = tuple(int(v) for v in sys.argv[2:5])
    main(sys.argv[1], seed)
