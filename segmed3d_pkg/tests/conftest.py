"""Shared pytest fixtures for the segmed3d test-suite.

Synthetic 3D volumes with embedded "lesions" are generated deterministically
so that all segmentation algorithms can be tested without external data.
"""

import numpy as np
import pytest


@pytest.fixture(scope="session")
def synthetic_volume():
    """A 3D volume (96, 96, 64) with two bright spherical lesions on a noisy
    background.  Returns ``(volume, ground_truth_mask)``.
    """
    rng = np.random.default_rng(seed=42)
    shape = (96, 96, 64)
    vol = rng.normal(loc=100.0, scale=10.0, size=shape).astype(np.float32)

    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    # Lesion 1 — sphere centred at (40, 45, 32), radius 12
    d2_1 = (x - 45) ** 2 + (y - 40) ** 2 + (z - 32) ** 2
    mask1 = d2_1 <= 12 ** 2
    vol[mask1] = rng.normal(loc=220.0, scale=8.0, size=mask1.sum()).astype(np.float32)

    # Lesion 2 — sphere centred at (60, 70, 28), radius 8
    d2_2 = (x - 70) ** 2 + (y - 60) ** 2 + (z - 28) ** 2
    mask2 = d2_2 <= 8 ** 2
    vol[mask2] = rng.normal(loc=200.0, scale=8.0, size=mask2.sum()).astype(np.float32)

    gt = (mask1 | mask2).astype(np.uint8)
    return vol, gt


@pytest.fixture(scope="session")
def affine_identity():
    """A 4x4 identity affine matrix."""
    return np.eye(4, dtype=np.float32)
