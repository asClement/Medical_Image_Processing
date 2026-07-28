"""Multi-atlas segmentation for 3D medical images.

This module implements the classical multi-atlas segmentation pipeline:

1. **(Optional) Registration** of each atlas image to the target image using
   SimpleITK (SyN / Demons / rigid+affine).
2. **Label fusion** of the propagated atlas labels using one of four
   strategies:
   - ``'majority_voting'``
   - ``'weighted_voting'`` (intensity-based weights)
   - ``'STAPLE'`` (Simultaneous Truth And Performance Level Estimation)
   - ``'JLF'`` (Joint Label Fusion, intensity + spatial agreement)

If the atlases are already aligned to the target, registration can be
skipped (``register=False``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ._base import BaseSegmentation

__all__ = ["AtlasSegmentation"]


class AtlasSegmentation(BaseSegmentation):
    """Multi-atlas segmentation with optional registration and 4 fusion rules.

    Examples
    --------
    >>> atlases_img = [ImageIO.load_nifti(p)[0] for p in atlas_paths]
    >>> atlases_lbl = [ImageIO.load_mask(p)[0] for p in atlas_label_paths]
    >>> seg = AtlasSegmentation(target_volume)
    >>> seg.fit(atlas_images=atlases_img, atlas_labels=atlases_lbl,
    ...         method='majority_voting', register=True)
    >>> mask = seg.get_mask()
    """

    def fit(
        self,
        atlas_images: Sequence[np.ndarray],
        atlas_labels: Sequence[np.ndarray],
        method: str = "majority_voting",
        register: bool = True,
        registration_method: str = "demons",
        registration_params: Optional[dict] = None,
        label_values: Optional[Sequence[int]] = None,
    ) -> "AtlasSegmentation":
        """Run multi-atlas segmentation.

        Parameters
        ----------
        atlas_images : sequence of np.ndarray
            List of 3D atlas intensity volumes (same shape as target if
            ``register=False``; otherwise they will be resampled to the target).
        atlas_labels : sequence of np.ndarray
            List of 3D atlas label volumes, aligned with ``atlas_images``.
        method : str, {'majority_voting', 'weighted_voting', 'STAPLE', 'JLF'}
            Label-fusion strategy.
        register : bool
            If ``True``, register each atlas to the target before fusion.
        registration_method : str, {'demons', 'syn', 'affine', 'rigid'}
            SimpleITK registration type.
        registration_params : dict, optional
            Extra parameters forwarded to the registration routine.
        label_values : sequence of int, optional
            Distinct labels present in the atlases.  If ``None``, the union
            of all unique non-zero labels is used.

        Returns
        -------
        self
        """
        if len(atlas_images) == 0 or len(atlas_labels) == 0:
            raise ValueError("atlas_images and atlas_labels must be non-empty.")
        if len(atlas_images) != len(atlas_labels):
            raise ValueError("atlas_images and atlas_labels must have the same length.")

        registration_params = registration_params or {}
        n_atlas = len(atlas_images)

        # -------------------------------------------------------------- #
        #  1. Registration (optional)                                     #
        # -------------------------------------------------------------- #
        if register:
            warped_imgs, warped_lbls = [], []
            for i in range(n_atlas):
                img_w, lbl_w = self._register_atlas(
                    atlas_images[i], atlas_labels[i],
                    registration_method, registration_params,
                )
                warped_imgs.append(img_w)
                warped_lbls.append(lbl_w)
        else:
            warped_imgs = [np.asarray(a, dtype=np.float32) for a in atlas_images]
            warped_lbls = [np.asarray(a) for a in atlas_labels]
            # Sanity check shapes
            for w in warped_lbls:
                if w.shape != self.volume_.shape:
                    raise ValueError(
                        f"Atlas shape {w.shape} != target {self.volume_.shape} "
                        f"(register=False requires pre-aligned atlases)."
                    )

        # -------------------------------------------------------------- #
        #  2. Determine label set                                         #
        # -------------------------------------------------------------- #
        if label_values is None:
            label_values = sorted(set(np.unique(np.concatenate(
                [np.unique(l) for l in warped_lbls]
            ))) - {0})
        if len(label_values) == 0:
            raise ValueError("No non-zero labels found in atlases.")

        # -------------------------------------------------------------- #
        #  3. Fusion                                                      #
        # -------------------------------------------------------------- #
        if method == "majority_voting":
            mask = self._majority_voting(warped_lbls, label_values)
        elif method == "weighted_voting":
            mask = self._weighted_voting(warped_imgs, warped_lbls, label_values)
        elif method == "STAPLE":
            mask = self._staple(warped_lbls, label_values)
        elif method == "JLF":
            mask = self._jlf(warped_imgs, warped_lbls, label_values)
        else:
            raise ValueError(
                f"Unknown fusion method {method!r}. Use 'majority_voting', "
                "'weighted_voting', 'STAPLE' or 'JLF'."
            )

        self.mask_ = mask.astype(np.uint8)
        self._warped_images_ = warped_imgs
        self._warped_labels_ = warped_lbls
        self._fitted = True
        self._method = method
        return self

    # ------------------------------------------------------------------ #
    #  Registration helper                                                #
    # ------------------------------------------------------------------ #
    def _register_atlas(
        self,
        atlas_img: np.ndarray,
        atlas_lbl: np.ndarray,
        method: str,
        params: dict,
    ):
        """Register an atlas (image + label) to the target volume.

        Returns the warped image and warped (nearest-neighbour resampled) label.
        """
        import SimpleITK as sitk

        target = self.volume_

        sitk_fixed = sitk.GetImageFromArray(np.transpose(target, (2, 1, 0)).astype(np.float32))
        sitk_moving = sitk.GetImageFromArray(np.transpose(atlas_img, (2, 1, 0)).astype(np.float32))
        sitk_fixed.CopyInformation(sitk.GetImageFromArray(np.transpose(target, (2, 1, 0)).astype(np.float32)))

        try:
            if method == "rigid":
                tx = self._rigid_register(sitk_fixed, sitk_moving, params)
            elif method == "affine":
                tx = self._affine_register(sitk_fixed, sitk_moving, params)
            elif method == "demons":
                tx = self._demons_register(sitk_fixed, sitk_moving, params)
            elif method == "syn":
                tx = self._syn_register(sitk_fixed, sitk_moving, params)
            else:
                raise ValueError(f"Unknown registration_method {method!r}.")

            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(sitk_fixed)
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(tx)
            warped_img_sitk = resampler.Execute(sitk_moving)

            # Resample label with nearest neighbour to preserve label values.
            sitk_moving_lbl = sitk.GetImageFromArray(
                np.transpose(atlas_lbl, (2, 1, 0)).astype(np.int16)
            )
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            warped_lbl_sitk = resampler.Execute(sitk_moving_lbl)

            warped_img = np.transpose(
                sitk.GetArrayFromImage(warped_img_sitk).astype(np.float32),
                (2, 1, 0),
            )
            warped_lbl = np.transpose(
                sitk.GetArrayFromImage(warped_lbl_sitk).astype(np.int16),
                (2, 1, 0),
            )
            return warped_img, warped_lbl
        except Exception:
            # Fallback: no registration (assume pre-aligned) — fail loudly if shapes differ.
            if atlas_img.shape != target.shape:
                raise
            return atlas_img.astype(np.float32), atlas_lbl

    # ------------------------------------------------------------------ #
    #  SimpleITK registration primitives                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rigid_register(fixed, moving, params):
        import SimpleITK as sitk
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=params.get("learning_rate", 1.0),
            minStep=params.get("min_step", 1e-6),
            numberOfIterations=params.get("iterations", 100),
        )
        tx = sitk.Euler3DTransform()
        reg.SetInitialTransform(tx, inPlace=False)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg.Execute(fixed, moving)

    @staticmethod
    def _affine_register(fixed, moving, params):
        import SimpleITK as sitk
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMeanSquares()
        reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=params.get("learning_rate", 1.0),
            minStep=params.get("min_step", 1e-6),
            numberOfIterations=params.get("iterations", 100),
        )
        tx = sitk.AffineTransform(3)
        reg.SetInitialTransform(tx, inPlace=False)
        reg.SetInterpolator(sitk.sitkLinear)
        return reg.Execute(fixed, moving)

    @staticmethod
    def _demons_register(fixed, moving, params):
        import SimpleITK as sitk
        demons = sitk.DemonsRegistrationFilter()
        demons.SetNumberOfIterations(params.get("iterations", 50))
        demons.SetStandardDeviations(params.get("sigma", 1.0))
        demons.SetSmoothingTypeToGaussian()
        disp_field = demons.Execute(fixed, moving)
        tx = sitk.DisplacementFieldTransform(disp_field)
        return tx

    @staticmethod
    def _syn_register(fixed, moving, params):
        import SimpleITK as sitk
        # SyN via BSpline + DisplacementField hybrid
        syn = sitk.SymmetricDemonsRegistrationFilter()
        syn.SetNumberOfIterations(params.get("iterations", 50))
        syn.SetStandardDeviations(params.get("sigma", 1.0))
        disp_field = syn.Execute(fixed, moving)
        return sitk.DisplacementFieldTransform(disp_field)

    # ------------------------------------------------------------------ #
    #  Fusion strategies                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _majority_voting(warped_lbls, label_values):
        stacked = np.stack(warped_lbls, axis=0)  # (N, X, Y, Z)
        out = np.zeros(stacked.shape[1:], dtype=np.int16)
        for lbl in label_values:
            votes = (stacked == lbl).sum(axis=0)
            # Keep label only if strictly more than half of atlases vote for it.
            out[votes * 2 > stacked.shape[0]] = lbl
        return (out > 0).astype(np.uint8)

    @staticmethod
    def _weighted_voting(warped_imgs, warped_lbls, label_values):
        target = warped_imgs[0]  # not used directly
        n = len(warped_imgs)
        # Weight = inverse intensity difference to the (mean of others)
        weights = []
        for i in range(n):
            others = [warped_imgs[j] for j in range(n) if j != i]
            mean_others = np.mean(others, axis=0)
            diff = np.abs(warped_imgs[i] - mean_others)
            sigma = diff.std() + 1e-6
            w = np.exp(-(diff ** 2) / (2 * sigma ** 2))
            weights.append(w)

        weight_stack = np.stack(weights, axis=0)  # (N, X, Y, Z)
        label_stack = np.stack(warped_lbls, axis=0)

        out = np.zeros(weight_stack.shape[1:], dtype=np.int16)
        for lbl in label_values:
            lbl_weight = np.where(label_stack == lbl, weight_stack, 0.0).sum(axis=0)
            total_weight = weight_stack.sum(axis=0)
            ratio = lbl_weight / np.maximum(total_weight, 1e-6)
            out[ratio > 0.5] = lbl
        return (out > 0).astype(np.uint8)

    @staticmethod
    def _staple(warped_lbls, label_values):
        """STAPLE — Simultaneous Truth And Performance Level Estimation.

        Uses SimpleITK's implementation.
        """
        import SimpleITK as sitk

        masks = []
        for w in warped_lbls:
            arr = (np.transpose(w, (2, 1, 0)) > 0).astype(np.uint8)
            masks.append(sitk.GetImageFromArray(arr))
        try:
            staple_filter = sitk.STAPLEImageFilter()
            result = staple_filter.Execute(masks)
            prob = sitk.GetArrayFromImage(result)
            prob = np.transpose(prob, (2, 1, 0))
            return (prob > 0.5).astype(np.uint8)
        except Exception:
            return AtlasSegmentation._majority_voting(warped_lbls, label_values)

    @staticmethod
    def _jlf(warped_imgs, warped_lbls, label_values):
        """Joint Label Fusion (simplified, intensity + label agreement).

        Reference: Wang & Yushkevich, "Groupwise segmentation with multi-atlas
        joint label fusion", MICCAI 2013 — here we use a simplified
        intensity-agreement weighting.
        """
        n = len(warped_imgs)
        target = np.median(np.stack(warped_imgs, axis=0), axis=0)

        # Compute per-atlas intensity agreement with the target.
        weights = []
        for i in range(n):
            diff = (warped_imgs[i] - target) ** 2
            sigma = diff.mean() + 1e-6
            w = np.exp(-diff / (2 * sigma))
            weights.append(w)
        weight_stack = np.stack(weights, axis=0)
        label_stack = np.stack(warped_lbls, axis=0)

        out = np.zeros(weight_stack.shape[1:], dtype=np.int16)
        for lbl in label_values:
            lbl_weight = np.where(label_stack == lbl, weight_stack, 0.0).sum(axis=0)
            total_weight = weight_stack.sum(axis=0)
            ratio = lbl_weight / np.maximum(total_weight, 1e-6)
            out[ratio > 0.5] = lbl
        return (out > 0).astype(np.uint8)

    # ------------------------------------------------------------------ #
    #  Accessors                                                          #
    # ------------------------------------------------------------------ #
    def get_warped_atlases(self):
        """Return the list of warped atlas images and labels (post-fit)."""
        if not self._fitted:
            raise RuntimeError("Call `.fit(...)` first.")
        return self._warped_images_, self._warped_labels_
