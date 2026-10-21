"""Field maps of the CFD slices for the paper, drawn with matplotlib.

The slices are resampled with pyvista onto a regular grid in the plane of the
map (finer than the smallest cell), then drawn with imshow. Axes, tick labels
and colour bars are therefore vector text at the printed size, with the fonts
of the line plots, and the colour maps are perceptually uniform (viridis family,
no jet/turbo). Each figure is built at the text width of the manuscript, with
the maps stacked one per row and the panel tags, (a), (b), ..., below each map,
as in the other figures.

Used by methodology_fields_fig.py (Fig. 3), clustering_fig.py
(Fig. 9), NN.py (Fig. 11), plot_fields.py (Figs. 12, 14, 15) and
plot_error_field.py (Fig. 13).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib import patheffects

ROTOR_DIAMETER = 126.0
HUB_HEIGHT = 90.0
ROTOR_RADIUS = 63.0

TEXTWIDTH_IN = 468.3324 / 72.27      # \textwidth of cas-sc
STYLE = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.top": False, "ytick.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.visible": False, "ytick.minor.visible": False,
    "savefig.dpi": 600,
    "savefig.bbox": None,   # keep the exact text width (scienceplots sets "tight")
}
TAG_SIZE = 9          # panel tags print at the size of the LaTeX ones

# layout (inches): left margin for the z/D label, right margin for the colour bar
LEFT, RIGHT_SHARED, RIGHT_ROW = 0.45, 0.75, 0.62
CBAR_GAP, CBAR_WIDTH = 0.08, 0.09
TOP = 0.2             # room for the colour-bar title of the first row
ROW_GAP = 0.32        # between rows: holds the tag of the row above
BOTTOM = 0.6          # tick labels, x/D label and the tag of the last row


def plane_cells(mesh, plane):
    """Cells of the y-normal (plane="xz", y = 0) or z-normal (plane="xy", hub
    height) slice. The training meshes hold both planes, which intersect along
    the rotor axis; probing them together would mix the two."""
    if plane == "xz":
        on_plane = np.isclose(mesh.points[:, 1], 0.0, atol=1e-6)
    else:
        on_plane = np.isclose(mesh.points[:, 2], HUB_HEIGHT, atol=1e-3)
    if on_plane.all():
        return mesh
    return mesh.extract_points(on_plane, adjacent_cells=False, include_cells=True)


def resample(mesh, scalars, x_range, v_range, plane="xz", spacing_d=0.008, categorical=False):
    """Values of `scalars` on a regular grid of the plane (coordinates in rotor
    diameters, mesh in metres). Returns an (nv, nx) array, NaN outside the mesh.
    `scalars` is an array name or a function mesh -> array (e.g. |U|)."""
    mesh = plane_cells(mesh, plane).copy()
    if callable(scalars):
        values = scalars(mesh)
        name = "_map_values"
        if len(values) == mesh.n_cells:
            mesh.cell_data[name] = values
        else:
            mesh.point_data[name] = values
    else:
        name = scalars
    # probe in a z-normal plane: the probe filter misses most points of a
    # y-normal grid, so the xz slice is first laid flat as (x, z, 0)
    points = mesh.points.copy()
    if plane == "xz":
        mesh.points = np.column_stack([points[:, 0], points[:, 2], np.zeros(len(points))])
    else:
        mesh.points = np.column_stack([points[:, 0], points[:, 1], np.zeros(len(points))])
    dx = spacing_d * ROTOR_DIAMETER
    nx = int(round((x_range[1] - x_range[0]) / spacing_d)) + 1
    nv = int(round((v_range[1] - v_range[0]) / spacing_d)) + 1
    grid = pv.ImageData(dimensions=(nx, nv, 1), spacing=(dx, dx, 1.0),
                        origin=(x_range[0] * ROTOR_DIAMETER, v_range[0] * ROTOR_DIAMETER, 0.0))
    sampled = grid.sample(mesh, categorical=categorical)
    values = np.asarray(sampled[name], dtype=float)
    valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    values[~valid] = np.nan
    return values.reshape(nv, nx)


def max_above(values, v_range, v_from=0.2):
    """Maximum of a resampled map above v_from (z/D): the modelled k of the RANS
    models peaks next to the ground, and a scale set on that peak would leave
    the wake unreadable."""
    rows = np.linspace(v_range[0], v_range[1], values.shape[0]) >= v_from
    return float(np.nanmax(values[rows]))


def velocity_magnitude(field="U"):
    return lambda mesh: np.linalg.norm(np.asarray(mesh[field]), axis=1)


def _draw_rotor(ax, plane):
    lo = (HUB_HEIGHT - ROTOR_RADIUS) / ROTOR_DIAMETER if plane == "xz" else -0.5
    hi = (HUB_HEIGHT + ROTOR_RADIUS) / ROTOR_DIAMETER if plane == "xz" else 0.5
    # white halo: the rotor stays visible on the dark end of the colour maps
    halo = [patheffects.withStroke(linewidth=2.4, foreground="white")]
    ax.plot([0, 0], [lo, hi], color="black", lw=1.2, solid_capstyle="butt", zorder=5, path_effects=halo)
    hub = HUB_HEIGHT / ROTOR_DIAMETER if plane == "xz" else 0.0
    ax.plot([0], [hub], "o", color="black", ms=2.2, mec="white", mew=0.5, zorder=6)


def _figure(n_rows, x_range, v_range, shared, right=None):
    right = right if right is not None else (RIGHT_SHARED if shared else RIGHT_ROW)
    map_w = TEXTWIDTH_IN - LEFT - CBAR_GAP - CBAR_WIDTH - right
    map_h = map_w * (v_range[1] - v_range[0]) / (x_range[1] - x_range[0])
    height = TOP + n_rows * map_h + (n_rows - 1) * ROW_GAP + BOTTOM
    fig = plt.figure(figsize=(TEXTWIDTH_IN, height))
    axes = []
    for i in range(n_rows):
        bottom = BOTTOM + (n_rows - 1 - i) * (map_h + ROW_GAP)
        axes.append(fig.add_axes((LEFT / TEXTWIDTH_IN, bottom / height, map_w / TEXTWIDTH_IN, map_h / height)))
    return fig, axes, (map_w, map_h, height)


def _format_axes(ax, x_range, v_range, plane, last):
    ax.set_xlim(*x_range)
    ax.set_ylim(*v_range)
    ax.set_xticks(np.arange(x_range[0], x_range[1] + 1e-9, 5))
    step = 1 if v_range[1] - v_range[0] <= 5 else 2
    ax.set_yticks(np.arange(np.ceil(v_range[0]), v_range[1] + 1e-9, step))
    ax.set_ylabel("z/D" if plane == "xz" else "y/D", labelpad=2)
    ax.tick_params(length=2.5, pad=1.5)
    if last:
        ax.set_xlabel("x/D", labelpad=1)
    else:
        ax.tick_params(labelbottom=False)


def _tag(fig, ax, tag, last, height):
    if not tag:
        return
    box = ax.get_position()
    below = 0.44 if last else ROW_GAP / 2   # inches under the axes (tick labels on the last row)
    fig.text(box.x0 + box.width / 2, box.y0 - below / height, f"({tag})",
             ha="center", va="center", fontsize=TAG_SIZE)


def map_stack(panels, output, x_range=(-5, 15), v_range=(0, 2.5), plane="xz",
              shared_colorbar=None, overlay=None):
    """Stack one map per row, full text width, and save it as vector PDF + PNG.

    panels: list of dicts with keys
        values  (nv, nx) array from resample()
        cmap, clim, cbar_label   (per-row colour bar unless shared_colorbar)
        extend  optional "max"/"min"/"both": arrow for values beyond clim
        tag     panel letter, drawn below the map
        title   optional label written in the upper-left corner of the map
        contours optional (values, levels) drawn as thin white isolines
    shared_colorbar: dict(cmap, clim, label) for a single bar spanning all rows.
    overlay: optional function(ax, panel) called after drawing each map.
    """
    with plt.rc_context(STYLE):
        fig, axes, (map_w, map_h, height) = _figure(len(panels), x_range, v_range, shared_colorbar is not None)
        extent = (x_range[0], x_range[1], v_range[0], v_range[1])
        images = []
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            cmap = shared_colorbar["cmap"] if shared_colorbar else panel["cmap"]
            clim = shared_colorbar["clim"] if shared_colorbar else panel["clim"]
            cmap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap
            cmap.set_bad("0.85")
            scale = {"norm": panel["norm"]} if panel.get("norm") is not None else {"vmin": clim[0], "vmax": clim[1]}
            image = ax.imshow(panel["values"], origin="lower", extent=extent, cmap=cmap, **scale,
                              interpolation="nearest", aspect="equal", rasterized=True)
            images.append(image)
            if panel.get("contours") is not None:
                values, levels = panel["contours"]
                xs = np.linspace(x_range[0], x_range[1], values.shape[1])
                vs = np.linspace(v_range[0], v_range[1], values.shape[0])
                ax.contour(xs, vs, values, levels=levels, colors="white", linewidths=0.4)
            _draw_rotor(ax, plane)
            if panel.get("title"):
                ax.text(0.006, 0.94, panel["title"], transform=ax.transAxes, ha="left", va="top", fontsize=8,
                        bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=7)
            if overlay is not None:
                overlay(ax, panel)
            last = i == len(panels) - 1
            _format_axes(ax, x_range, v_range, plane, last)
            _tag(fig, ax, panel.get("tag"), last, height)
            if not shared_colorbar and panel.get("cbar_label"):
                box = ax.get_position()
                cax = fig.add_axes((box.x1 + CBAR_GAP / TEXTWIDTH_IN, box.y0, CBAR_WIDTH / TEXTWIDTH_IN, box.height))
                bar = fig.colorbar(image, cax=cax, extend=panel.get("extend", "neither"))
                bar.ax.tick_params(length=2, pad=1.5, labelsize=7)
                bar.locator = plt.MaxNLocator(nbins=4)
                bar.update_ticks()
                bar.outline.set_linewidth(0.5)
                cax.set_title(panel["cbar_label"], fontsize=8, pad=3)
        if shared_colorbar:
            top, bottom = axes[0].get_position(), axes[-1].get_position()
            cax = fig.add_axes((top.x1 + CBAR_GAP / TEXTWIDTH_IN, bottom.y0,
                                CBAR_WIDTH / TEXTWIDTH_IN, top.y1 - bottom.y0))
            bar = fig.colorbar(images[0], cax=cax, extend=shared_colorbar.get("extend", "neither"))
            bar.ax.tick_params(length=2, pad=1.5, labelsize=7)
            bar.outline.set_linewidth(0.5)
            bar.set_label(shared_colorbar["label"], fontsize=8, labelpad=3)
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".pdf"))
        fig.savefig(output.with_suffix(".png"), dpi=300)
        plt.close(fig)
