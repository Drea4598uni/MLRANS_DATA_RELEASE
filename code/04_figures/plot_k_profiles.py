"""Vertical profiles of the turbulent kinetic energy k behind the rotor: LES,
standard RANS and ML-RANS on the ~4.2M cells grid ("5M"), at the same sampling
stations as the velocity-deficit figures, for the three inflow velocities.

The LES k is the total turbulent kinetic energy, resolved plus subgrid-scale,
0.5 tr(UPrime2Mean) + 0.5 tr(R_sgs), recomputed from the tensors for every case
(the "k" array stored in the 8 m/s LES slice is a uniform field, not used, and the
15 m/s slice has none). For RANS and ML-RANS it is the modelled k of the k-omega
SST equations, the field shown in the k maps of Figs. 12, 14 and 15.
k is plotted as k/U_inf^2, with the hub-height inflow velocity of each case, the
same reference as Delta U/U_inf in the velocity-deficit figures.

Run from the package root:
    python code/04_figures/plot_k_profiles.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scienceplots  # noqa: F401  (registers the 'science' styles)
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc

plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'figure.dpi': 300,
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'lines.linewidth': 0.9,
    'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.top': False, 'ytick.right': False,
})

OUTPUT = Path("figures_output/ml_results/k_profiles_5m.png")

ML_LABEL = "ML-RANS 5M"
# sample line stations (x/D), the same as in the velocity-deficit figures
STATIONS = [0.5, 1, 2, 3, 4, 5, 6, 7]
LINE_TOP = 5 * cc.ROTOR_RADIUS
# the modelled k of both RANS models peaks close to the ground (by far more in
# ML-RANS); the axis range is set on the profiles above this height, just below
# the lower blade tip (z/D = 0.21), so that the wake is readable and the
# near-ground peak runs off the axis
Z_RANGE_FROM = 0.2


def les_tke(mesh):
    """Total LES k: resolved (UPrime2Mean) plus subgrid (R) part. Symmetric
    tensors are stored in VTK order (XX, YY, ZZ, XY, YZ, XZ)."""
    resolved = np.asarray(mesh["UPrime2Mean"])[:, :3].sum(axis=1)
    subgrid = np.asarray(mesh["turbulenceProperties:R"])[:, :3].sum(axis=1)
    return 0.5 * (resolved + subgrid)


def read_case(case):
    """y-normal plane of the LES, RANS and ML-RANS 5M results, with k/U_inf^2 on points."""
    u2 = cc.UREF[case] ** 2
    les = pv.read(cc.les_ref(case)[0][0])
    les["k_profile"] = les_tke(les) / u2
    rans = pv.read(cc.rans_ref_meshes(case)[0])
    rans["k_profile"] = np.asarray(rans["k"]) / u2
    ml = pv.read(cc.solver_meshes(case, ML_LABEL)[0]).cell_data_to_point_data()
    ml["k_profile"] = np.asarray(ml["k"]) / u2
    return {"LES": les, "RANS": rans, ML_LABEL: ml}


def sample_profile(mesh, station):
    """k/U_inf^2 and z/D along the vertical line at x/D = station; invalid samples
    (outside the mesh) are dropped."""
    x = station * cc.ROTOR_DIAMETER
    sampled = pv.Line((x, 0, 0), (x, 0, LINE_TOP), resolution=1000).sample(mesh)
    z_over_d = sampled.points[:, 2] / cc.ROTOR_DIAMETER
    valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    return np.asarray(sampled["k_profile"])[valid], z_over_d[valid]


def plot_profile(ax, label, k, z_over_d):
    if label == "LES":
        ax.scatter(k[::20], z_over_d[::20], label=label, marker="x", color="black", s=6, linewidths=0.8)
    elif label == "RANS":
        ax.plot(k, z_over_d, label=label, color="black", linestyle="--")
    else:
        ax.plot(k, z_over_d, label=label, color="blue", linestyle="-")


def main():
    fig, axes = plt.subplots(len(cc.CASES), len(STATIONS), figsize=(6.5, 7.2), sharey=True)
    for row, (tag, case) in zip(axes, zip("abc", cc.CASES)):
        meshes = read_case(case)
        k_max = 0.0
        for ax, station in zip(row, STATIONS):
            for label in ("LES", "RANS", ML_LABEL):
                k, z_over_d = sample_profile(meshes[label], station)
                plot_profile(ax, label, k, z_over_d)
                k_max = max(k_max, k[z_over_d >= Z_RANGE_FROM].max())
            ax.set_title(f"x/D = {station:g}", pad=3)
        for ax in row:
            ax.set_xlim(0, 1.05 * k_max)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10]))
            ax.set_xlabel(r"$k/U_\infty^2$")
        row[0].set_ylabel("z/D")
        row[0].annotate(rf"({tag}) $U_\infty = {cc.UREF[case]:g}$ m/s", xy=(0, 1), xycoords="axes fraction",
                        xytext=(-22, 16), textcoords="offset points", ha="left", va="bottom", fontsize=8)
    axes[0, 0].set_ylim(0, LINE_TOP / cc.ROTOR_DIAMETER * 1.02)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, [cc.display_label(label) for label in labels], loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.2, w_pad=0.6)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cc.save_figure(OUTPUT, fig)
    plt.close(fig)


if __name__ == "__main__":
    main()
