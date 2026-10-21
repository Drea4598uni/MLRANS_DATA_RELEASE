"""Rotor-disk metrics from the cross-stream planes x = const (x/D = 2-7):
available power of a downstream rotor, error with respect to the LES,
100 (<Ux^3>_disk / <Ux^3>_disk,LES - 1), and rotor-averaged turbulence
intensity sqrt(2/3 <k>_disk)/U_inf, for the three inflow velocities.

Data: data/08_cross_planes/disk_metrics.csv, produced on the cluster with
code/05_cross_planes (see data/08_cross_planes/MANIFEST.md). LES k is the resolved part
only (no mean SGS field was stored).

Run from the package root:
    python code/04_figures/plot_cross_planes.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc  # noqa: E402

DATA = Path("data/08_cross_planes/disk_metrics.csv")
OUTPUT = Path("figures_output/ml_results/cross_plane_metrics.png")
STYLE = {
    "text.usetex": False, "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman No9 L", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "lines.linewidth": 0.9, "lines.markersize": 3,
    "axes.spines.top": False, "axes.spines.right": False,
}
# model folder -> (legend label, colour, line style, marker), as in the deficit figures
MODELS = {
    "RANS_30.8M": ("RANS 30.8M", "black", "--", "^"),
    "ML-RANS_15.9M": ("ML-RANS 15.9M", "red", "-", "o"),
    "ML-RANS_9.1M": ("ML-RANS 9.1M", "green", "-", "o"),
    "ML-RANS_4.2M": ("ML-RANS 4.2M", "blue", "-", "o"),
    "ML-RANS_4.2M_noClustering": ("ML-RANS 4.2M no clustering", "magenta", "-", "o"),
}
CASES = {"8ms": "8", "11ms": "11.4", "15ms": "15"}


def main():
    data = pd.read_csv(DATA)
    with plt.rc_context(STYLE):
        # same y range in every column, so that the three velocities compare directly
        fig, axes = plt.subplots(2, 3, figsize=(6.5, 4.4), sharex=True, sharey="row")
        for col, (case, speed) in enumerate(CASES.items()):
            sub = data[data["case"] == case]
            top, bottom = axes[0, col], axes[1, col]
            top.axhline(0.0, color="0.6", lw=0.6)
            for model, (label, color, ls, marker) in MODELS.items():
                m = sub[sub["model"] == model].sort_values("x_D")
                top.plot(m["x_D"], m["P_err_pct"], color=color, ls=ls, marker=marker, label=label)
                bottom.plot(m["x_D"], m["TI_rotor_pct"], color=color, ls=ls, marker=marker, label=label)
            les = sub[sub["model"] == "LES"].sort_values("x_D")
            bottom.plot(les["x_D"], les["TI_rotor_pct"], color="black", ls="none", marker="x", ms=4, label="LES")
            top.set_title(rf"$U_\infty = {speed}$ m/s")
            bottom.set_xlabel("x/D")
        axes[0, 0].set_ylim(-25, 50)   # data span -21 % ... +48 %
        axes[1, 0].set_ylim(0, 14)
        axes[0, 0].set_ylabel("Available-power error [%]")
        axes[1, 0].set_ylabel(r"Rotor-averaged TI [%]")
        handles, labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        cc.save_figure(OUTPUT, fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
