"""Visualization utilities for 3D volumes and masks.

Provides orthogonal slice views, mask overlays, surface plots and
histograms via :mod:`matplotlib`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["Visualizer"]


class Visualizer:
    """Static collection of plotting routines.

    All methods create a matplotlib figure and return it.  The caller is
    responsible for ``plt.show()`` / ``fig.savefig(...)``.
    """

    # ------------------------------------------------------------------ #
    #  Orthogonal slices                                                  #
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_3d_slices(
        volume: np.ndarray,
        mask: Optional[np.ndarray] = None,
        slice_indices: Optional[Tuple[int, int, int]] = None,
        cmap: str = "gray",
        mask_alpha: float = 0.4,
        mask_color: str = "red",
        figsize: Tuple[float, float] = (12, 4),
    ):
        """Plot the three orthogonal mid-slices (axial, coronal, sagittal).

        Parameters
        ----------
        volume : np.ndarray
            3D volume.
        mask : np.ndarray, optional
            Optional binary mask overlaid in colour.
        slice_indices : tuple of int, optional
            ``(i_axial, i_coronal, i_sagittal)``.  Defaults to mid-slices.
        cmap : str
            Colormap for the volume.
        mask_alpha : float
            Overlay transparency.
        mask_color : str
            Overlay colour.
        figsize : tuple
            Figure size.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        vol = np.asarray(volume, dtype=np.float32)
        if slice_indices is None:
            slice_indices = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        titles = ["Axial (X)", "Coronal (Y)", "Sagittal (Z)"]
        views = [
            vol[slice_indices[0], :, :],
            vol[:, slice_indices[1], :],
            vol[:, :, slice_indices[2]],
        ]
        mask_views = None
        if mask is not None:
            m = np.asarray(mask) > 0
            mask_views = [
                m[slice_indices[0], :, :],
                m[:, slice_indices[1], :],
                m[:, :, slice_indices[2]],
            ]

        for ax, im, t in zip(axes, views, titles):
            ax.imshow(im.T, cmap=cmap, origin="lower")
            if mask_views is not None:
                masked = np.ma.masked_where(mask_views[axes.tolist().index(ax)] == 0,
                                            mask_views[axes.tolist().index(ax)])
                ax.imshow(masked.T, cmap="autumn", alpha=mask_alpha, origin="lower")
            ax.set_title(t)
            ax.axis("off")

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  Overlay                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_overlay(
        volume: np.ndarray,
        mask: np.ndarray,
        slice_axis: int = 2,
        slice_index: Optional[int] = None,
        alpha: float = 0.4,
        cmap: str = "gray",
        mask_cmap: str = "autumn",
        figsize: Tuple[float, float] = (6, 6),
    ):
        """Overlay a mask on a single slice of the volume."""
        import matplotlib.pyplot as plt

        vol = np.asarray(volume, dtype=np.float32)
        m = np.asarray(mask) > 0
        if slice_index is None:
            slice_index = vol.shape[slice_axis] // 2

        if slice_axis == 0:
            v_slice, m_slice = vol[slice_index, :, :], m[slice_index, :, :]
        elif slice_axis == 1:
            v_slice, m_slice = vol[:, slice_index, :], m[:, slice_index, :]
        else:
            v_slice, m_slice = vol[:, :, slice_index], m[:, :, slice_index]

        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(v_slice.T, cmap=cmap, origin="lower")
        masked = np.ma.masked_where(m_slice == 0, m_slice)
        ax.imshow(masked.T, cmap=mask_cmap, alpha=alpha, origin="lower")
        ax.set_title(f"Axis {slice_axis} - slice {slice_index}")
        ax.axis("off")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  3D surface                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_3d_surface(
        mask: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        figsize: Tuple[float, float] = (8, 8),
        color: str = "gold",
    ):
        """Render the surface of a binary mask as a 3D mesh.

        Requires :mod:`skimage.measure.marching_cubes`.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from skimage.measure import marching_cubes

        m = (np.asarray(mask) > 0).astype(np.float32)
        if m.sum() == 0:
            raise ValueError("Cannot plot surface: mask is empty.")
        # Pad to ensure the surface is closed.
        padded = np.pad(m, 1, mode="constant", constant_values=0)
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing)

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        mesh = Poly3DCollection(verts[faces], alpha=0.7, facecolor=color,
                                edgecolor="k", linewidths=0.1)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, padded.shape[0])
        ax.set_ylim(0, padded.shape[1])
        ax.set_zlim(0, padded.shape[2])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D mask surface")
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------ #
    #  Histogram                                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def plot_histogram(
        volume: np.ndarray,
        bins: int = 100,
        mask: Optional[np.ndarray] = None,
        figsize: Tuple[float, float] = (8, 5),
    ):
        """Plot the intensity histogram of a volume (optionally restricted to a mask)."""
        import matplotlib.pyplot as plt

        vol = np.asarray(volume, dtype=np.float32).ravel()
        if mask is not None:
            m = np.asarray(mask) > 0
            vol = vol[m.ravel()]

        fig, ax = plt.subplots(figsize=figsize)
        ax.hist(vol, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Count")
        ax.set_title("Intensity histogram")
        ax.set_yscale("log")
        plt.tight_layout()
        return fig
