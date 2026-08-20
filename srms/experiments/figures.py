"""Per-run figures: ground truth, learned field, signed error — publication quality.

Each call writes **two** files: a labelled version and a bare one carrying identical pixels with
every axis, title, tick and colourbar stripped, for a paper where the caption and annotation are set
in LaTeX. Both are vector PDF plus high-resolution PNG.

Level sets are drawn on the two field panels. They are what makes a time-to-go field readable: evenly
spaced contours mean the gradient magnitude is correct, and a contour bending around an obstacle is
the detour the field has learned. Comparing contour *spacing* between the ground-truth and learned
panels shows gradient error, which the colour map alone hides.

Manifolds whose scoring grid is 2-D (torus, sphere, Poincaré ball) are drawn in their own chart.
SO(3)'s grid is 3-D geodesic-polar about the identity, so it is drawn as a slice at the middle
colatitude: rotation angle against azimuth, source at angle 0.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SLICE_EXTENT = (-180.0, 180.0, 0.0, 180.0)  # SO(3) slice: azimuth (deg) x rotation angle (deg)
_CONTOURS = 12
_DPI = 300

# Times New Roman where available, with the metric-compatible Nimbus Roman as the fallback so a
# machine without the Microsoft fonts still produces the same layout.
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.linewidth": 0.8,
        "savefig.bbox": "tight",
    }
)


def _as_image(field: np.ndarray, shape: tuple[int, ...]):
    """Reshape a raveled field to a 2-D image, slicing SO(3)'s 3-D polar grid at mid-colatitude.

    Args:
        field: Raveled field, [prod(shape)].
        shape: Per-axis grid shape, 2-D or 3-D.

    Returns:
        ``(image, extent_override)``; ``extent_override`` is None for a genuinely 2-D chart.
    """
    if len(shape) == 2:
        return field.reshape(shape), None
    volume = field.reshape(shape)
    return volume[:, volume.shape[1] // 2, :], _SLICE_EXTENT


def _panels(truth_img, pred_img, error_img, vmax, error_clip, reference):
    """Panel specs: (title, image, colormap, vmin, vmax, draw_contours)."""
    return (
        (f"{reference}", truth_img, "viridis", 0.0, vmax, True),
        ("learned $T = \\mathrm{base}\\cdot e^{g}$", pred_img, "viridis", 0.0, vmax, True),
        (f"error (learned $-$ {reference})", error_img, "bwr", -error_clip, error_clip, False),
    )


def _draw(ax, img, cmap, lo, hi, extent, contours: bool, levels):
    """Draw one panel and return its image handle; overlay level sets when asked."""
    handle = ax.imshow(img, origin="lower", extent=extent, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    if contours:
        rows, cols = img.shape
        mesh_x, mesh_y = np.meshgrid(
            np.linspace(extent[0], extent[1], cols), np.linspace(extent[2], extent[3], rows)
        )
        ax.contour(mesh_x, mesh_y, img, levels=levels, colors="white", linewidths=0.7, alpha=0.85)
    return handle


def save(
    env,
    out_path: str,
    truth,
    prediction,
    shape,
    mask=None,
    error_clip: float = 0.05,
    title: str = "",
    reference: str = "ground truth",
) -> None:
    """Write the labelled figure (``out_path``) and a bare twin (``*_bare.pdf``/``.png``).

    Args:
        env: Environment, read for ``render_extent``, ``axis_labels`` and ``render_marker_deg``.
        out_path: Destination path; the extension is replaced to emit both PDF and PNG.
        truth: Reference field, raveled.
        prediction: Learned field, raveled.
        shape: Per-axis grid shape.
        mask: Optional boolean array, True where a cell is outside the domain and is blanked.
        error_clip: Symmetric colour limit for the signed-error panel.
        title: Figure suptitle (labelled version only).
        reference: Name of the field in the first panel — the analytic geodesic with no obstacles,
            fast marching (the ground truth) with them.
    """
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if mask is not None:
        truth = np.where(mask, np.nan, truth)
        prediction = np.where(mask, np.nan, prediction)
    truth_img, override = _as_image(truth, shape)
    pred_img, _ = _as_image(prediction, shape)
    error_img = pred_img - truth_img
    extent = override if override is not None else env.render_extent
    vmax = float(np.nanmax(truth_img))
    levels = np.linspace(0.0, vmax, _CONTOURS + 2)[1:-1]  # shared, so spacing is comparable panel to panel
    marker = (0.0, 0.0) if override is not None else env.render_marker_deg()
    panels = _panels(truth_img, pred_img, error_img, vmax, error_clip, reference)
    stem = out_path.rsplit(".", 1)[0]

    for labelled in (True, False):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6) if labelled else (16.5, 4.2))
        for ax, (name, img, cmap, lo, hi, contours) in zip(axes, panels):
            handle = _draw(ax, img, cmap, lo, hi, extent, contours, levels)
            ax.plot(
                *marker, marker="*", markersize=19, color="#ffe14d", markeredgecolor="black", markeredgewidth=0.9
            )
            if labelled:
                ax.set_title(name, fontsize=13)
                ax.set_xlabel(env.axis_labels[0] if override is None else "azimuth (deg)", fontsize=11)
                ax.set_ylabel(env.axis_labels[1] if override is None else "rotation angle (deg)", fontsize=11)
                ax.tick_params(labelsize=9)
                fig.colorbar(handle, ax=ax, fraction=0.046, pad=0.02)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        if labelled and title:
            fig.suptitle(title, fontsize=14)
        fig.tight_layout()
        suffix = "" if labelled else "_bare"
        for ext in ("pdf", "png"):
            fig.savefig(f"{stem}{suffix}.{ext}", dpi=_DPI, bbox_inches="tight",
                        pad_inches=0.0 if not labelled else 0.05)
        plt.close(fig)
