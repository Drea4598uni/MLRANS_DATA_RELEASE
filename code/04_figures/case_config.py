"""Shared per-inflow-case configuration for the results figures.

Importing this module changes the working directory to the package root, so
all paths here are relative to it and the scripts can be launched from anywhere. The dataset was reorganised into three inflow-velocity cases
(8/11/15 m/s); this module centralises the (slightly inhomogeneous) file layout
so the individual plotting scripts stay clean.

Layout reminder
---------------
- solver (ML-RANS) results : data/06_solver_results/<case>/{y,z}Normal_<suffix>.vtp
      suffix in {15ml, 10ml, 5ml|5, noClustering}; note the 5M mesh at 11 m/s is
      named "_5" (without "ml").
- RANS reference           : data/02_rans_reference/{y,z}Normal_<case>.vtp
- LES reference            : 11ms -> data/01_les_reference/les_mesh_with_k.vtp (has k)
                             8ms  -> data/01_les_reference/8ms/yNormal.vtp     (its k is a
                                     uniform field, not used)
                             15ms -> data/01_les_reference/15ms/yNormal.vtp    (NO k)
                             all three carry UPrime2Mean and the SGS stress R,
                             from which plot_k_profiles.py computes the total k
- NN predictions (11 m/s)  : data/05_neural_network/ml_rans_clustering/...
- NN predictions no clust. : data/05_neural_network/ml_rans_no_clustering/...
- clustering mesh          : data/04_clustering/clustered_mesh_K7.vtu
"""

import os
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]   # package root
os.chdir(ROOT)
OUT = Path("figures_output")                  # every figure is written here
OUT.mkdir(exist_ok=True)
import numpy as np

# --- figure output (journal artwork requirements) --------------------------
# pyvista scenes are rendered in a window this many times larger than the
# original 1024x768 layout, with the render-window DPI raised by the same factor
# so text in points keeps its relative size. Pixel positions and the cube-axes
# label size (in pixels) must be multiplied by it explicitly. The PNG dpi is
# raised too, so the printed size (and the LaTeX trim values, in bp) stay put.
SCREENSHOT_SCALE = 3


def save_figure(path, fig=None, **kwargs):
    """Save a matplotlib figure as PNG and as vector PDF with the same stem."""
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.gcf()
    path = Path(path)
    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(path.with_suffix(".png"), **kwargs)
    fig.savefig(path.with_suffix(".pdf"), **kwargs)


def hires_plotter(window_size=(1024, 768)):
    """Off-screen pyvista plotter SCREENSHOT_SCALE times larger than window_size."""
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=[s * SCREENSHOT_SCALE for s in window_size])
    plotter.ren_win.SetDPI(72 * SCREENSHOT_SCALE)
    return plotter


def px(position):
    """Scale a pixel position laid out for the original window size."""
    return tuple(v * SCREENSHOT_SCALE for v in position)


def hires_screenshot(plotter, path, base_dpi=72):
    """Screenshot of a hires_plotter; base_dpi is the dpi the PNG was implicitly
    read at before (72 when pyvista writes no dpi)."""
    plotter.screenshot(str(path))
    dpi = base_dpi * SCREENSHOT_SCALE
    Image.open(path).save(path, dpi=(dpi, dpi))


# --- geometry -------------------------------------------------------------
ROTOR_DIAMETER = 126.0
ROTOR_RADIUS = 63.0

def scale_to_diameters(mesh):
    """Express coordinates in rotor diameters, snapping the ground (stored at
    z ~ -1e-15 m in some exports, printed as "-0.0D") to exactly zero."""
    points = mesh.points / ROTOR_DIAMETER
    points[np.abs(points[:, 2]) < 1e-9, 2] = 0.0
    mesh.points = points
    return mesh


def show_xz_bounds(plotter, bounds_args):
    """show_bounds that labels the nominal `bounds` rather than each mesh's own
    extent (which gave 1.2D on one panel and 1.3D on the next). Labels use %g
    (-5D, 0D, 1.25D, 2.5D): exact for the 1.25D mid tick and short enough that
    the z = 0 label does not overlap the x = -5 one (as in Fig. 9).
    Call it after the last add_mesh/add_text: adding an actor afterwards makes
    pyvista reset the axis ranges to the data extent."""
    args = dict(bounds_args)
    args.setdefault("axes_ranges", args["bounds"])
    args["fmt"] = "%gD"
    cube_axes = plotter.show_bounds(**args)
    # the label-to-axis gap is in pixels (VTK default 20): scale it with the
    # hires window, otherwise the z = 0 and x = -5 labels collide at the corner
    cube_axes.SetLabelOffset(20 * SCREENSHOT_SCALE)
    return cube_axes


# --- inflow cases ---------------------------------------------------------
CASES = ["8ms", "11ms", "15ms"]
PRINCIPAL = "11ms"
UREF = {"8ms": 8.0, "11ms": 11.4, "15ms": 15.0}  # hub-height inflow of the runs; "11ms" is only the folder name

# ML-RANS variants -> solver file suffix (label as used in the legends)
SOLVER_SUFFIX = {
    "ML-RANS 15M": ["15ml"],
    "ML-RANS 10M": ["10ml"],
    "ML-RANS 5M": ["5ml", "5"],          # 11 m/s uses "_5" (no "ml")
    "ML-RANS no clustering": ["noClustering"],
}

# --- prediction / clustering meshes (single, 11 m/s training) -------------
PRED_CLU = "data/05_neural_network/ml_rans_clustering/results/rans_with_predictions.vtu"
PRED_CLU_HISTORY = "data/05_neural_network/ml_rans_clustering/results/training_history_nut.csv"
PRED_CLU_METRICS = "data/05_neural_network/ml_rans_clustering/results/metrics_nut.json"
PRED_NOCLU = "data/05_neural_network/ml_rans_no_clustering/results/rans_with_predictions.vtu"
PRED_NOCLU_HISTORY = "data/05_neural_network/ml_rans_no_clustering/results/training_history_nut.csv"
PRED_NOCLU_METRICS = "data/05_neural_network/ml_rans_no_clustering/results/metrics_nut.json"

CLUSTER_DIR = Path("data/04_clustering")
CLUSTER_MESH = str(CLUSTER_DIR / "clustered_mesh_K7.vtu")        # carries the cluster "ID" field
CLUSTER_RANS_MESH = str(CLUSTER_DIR / "rans_mod.vtk")      # baseline RANS nut field
CLUSTER_ELBOW = str(CLUSTER_DIR / "elbow.csv")
CLUSTER_PCA_VARIANCE = str(CLUSTER_DIR / "pca_explained_variance_ratio.csv")
CLUSTER_PCA_IMPORTANCE = str(CLUSTER_DIR / "pca_importance.csv")


# legend labels: the paper names the grids after their cell count (Table 2);
# the keys above stay as they are because they name folders and files
DISPLAY_LABEL = {
    "ML-RANS 15M": "ML-RANS 15.9M",
    "ML-RANS 10M": "ML-RANS 9.1M",
    "ML-RANS 5M": "ML-RANS 4.2M",
    "ML-RANS no clustering": "ML-RANS 4.2M no clustering",
    "RANS": "RANS 30.8M",          # the standard RANS runs on the LES grid
}


def display_label(label):
    return DISPLAY_LABEL.get(label, label)


def first_existing(paths):
    """Return the first path that exists (handles the _5ml vs _5 naming)."""
    for path in paths:
        if Path(path).exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exist: {paths}")


def solver_meshes(case, label):
    """Return [yNormal, zNormal] solver result paths for an ML-RANS variant."""
    base = Path("data/06_solver_results") / case
    out = []
    for plane in ("yNormal", "zNormal"):
        candidates = [str(base / f"{plane}_{suffix}.vtp") for suffix in SOLVER_SUFFIX[label]]
        out.append(first_existing(candidates))
    return out


def rans_ref_meshes(case):
    """Return [yNormal, zNormal] RANS-reference paths for a case."""
    base = Path("data/02_rans_reference")
    return [str(base / f"yNormal_{case}.vtp"), str(base / f"zNormal_{case}.vtp")]


def les_ref(case):
    """Return (paths, velocity_field, has_k) for the LES reference of a case."""
    if case == "11ms":
        return (["data/01_les_reference/les_mesh_with_k.vtp"], "UMean", True)
    if case == "8ms":
        return (["data/01_les_reference/8ms/yNormal.vtp"], "UMean", True)
    if case == "15ms":
        return (["data/01_les_reference/15ms/yNormal.vtp"], "UMean", False)
    raise ValueError(f"Unknown case: {case}")
