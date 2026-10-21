import sys

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import scienceplots
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc

plt.style.use(['science', 'ieee'])
# --- SEEP2026 template: force Times New Roman to match the manuscript body ---
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False

OUTPUT_ROOT = Path("figures_output/ml_results")

rotor_diameter = cc.ROTOR_DIAMETER
rotor_radius = cc.ROTOR_RADIUS

ML_COLORS = {
    "ML-RANS 15M": "red",
    "ML-RANS 10M": "green",
    "ML-RANS 5M": "blue",
    "ML-RANS no clustering": "magenta",
}


# sample line stations (x/D); negative ones are upstream of the rotor
x = [0.5, 1, 2, 3, 4, 5, 6, 7]
# the integral metric only makes sense in the wake (downstream of the rotor)
x_metric = [i for i in x if i > 0]

# labels (and their order) used in the integral-difference metric plot
METRIC_LABELS = ["ML-RANS 15M", "ML-RANS 10M", "ML-RANS 5M", "ML-RANS no clustering", "RANS"]

comparison_groups = [
    ("ml_rans_15m_vs_rans_les", ["ML-RANS 15M", "RANS", "LES"]),
    ("ml_rans_10m_vs_rans_les", ["ML-RANS 10M", "RANS", "LES"]),
    ("ml_rans_5m_vs_rans_les", ["ML-RANS 5M", "RANS", "LES", "ML-RANS no clustering"]),
    ("ml_rans_no_clustering_vs_rans_les", ["ML-RANS no clustering", "RANS", "LES"]),
    ("ml_models_comparison", ["ML-RANS 15M", "ML-RANS 10M", "ML-RANS 5M", "ML-RANS no clustering", "RANS", "LES"]),
]


def interpolate_cell_fields_to_points(mesh, fields):
    fields_to_check = set(fields) | {"k"}
    if any(field in mesh.cell_data for field in fields_to_check):
        return mesh.cell_data_to_point_data()
    return mesh


def valid_sample_mask(sampled):
    if "vtkValidPointMask" not in sampled.array_names:
        return np.ones(sampled.n_points, dtype=bool)
    return np.asarray(sampled["vtkValidPointMask"]).astype(bool)


def interpolate_invalid_samples(values, z_over_d, valid):
    values = np.asarray(values, dtype=float)
    if valid.all():
        return values
    if valid.sum() < 2:
        return values
    return np.interp(z_over_d, z_over_d[valid], values[valid])


def velocity_magnitude(sampled, velocity_field):
    velocity = np.asarray(sampled[velocity_field])
    if velocity.ndim == 2:
        velocity = np.linalg.norm(velocity, axis=1)
    return velocity


def velocity_deficit_profile(sampled, velocity_field, uref):
    """Interpolated (over invalid samples) velocity-deficit profile and z/D grid."""
    deficit = 1 - velocity_magnitude(sampled, velocity_field) / uref
    z_over_d = sampled.points[:, 2] / rotor_diameter
    valid = valid_sample_mask(sampled)
    deficit = interpolate_invalid_samples(deficit, z_over_d, valid)
    return deficit, z_over_d


def plot_velocity_deficit(label, sampled, velocity_field, uref, ax=None, marker_size=18):
    ax = ax if ax is not None else plt.gca()
    velocity = velocity_magnitude(sampled, velocity_field)
    velocity_deficit = 1 - velocity / uref
    z_over_d = sampled.points[:, 2] / rotor_diameter
    valid = valid_sample_mask(sampled)

    if label == "LES":
        ax.scatter(velocity_deficit[valid][::20], z_over_d[valid][::20], label=label, marker="x", color="black", s=marker_size, linewidths=0.8)
    elif label == "RANS":
        ax.plot(velocity_deficit[valid], z_over_d[valid], label=label, color="black", linewidth=1.2, linestyle="--")
    else:
        velocity_deficit = interpolate_invalid_samples(velocity_deficit, z_over_d, valid)
        ax.plot(velocity_deficit, z_over_d, label=label, color=ML_COLORS[label], linestyle="-")


def plot_k_deficit(label, sampled):
    z_over_d = sampled.points[:, 2] / rotor_diameter
    valid = valid_sample_mask(sampled)

    if label == "LES":
        plt.scatter(sampled["k"][valid][::20], z_over_d[valid][::20], label=label, marker="x", facecolor="black", s=18, linewidths=0.8)
    elif label == "RANS":
        plt.scatter(sampled["k"][valid][::20], z_over_d[valid][::20], label=label, marker="^", facecolor="none", edgecolors="black", s=18)
    else:
        k = interpolate_invalid_samples(sampled["k"], z_over_d, valid)
        plt.plot(k, z_over_d, label=label, color=ML_COLORS[label], linestyle="-")


def build_case_simulations(case):
    """Read every mesh needed for a given inflow case and pre-interpolate cell data."""
    les_paths, les_field, les_has_k = cc.les_ref(case)
    raw = [
        ("ML-RANS 15M", cc.solver_meshes(case, "ML-RANS 15M"), "U"),
        ("ML-RANS 10M", cc.solver_meshes(case, "ML-RANS 10M"), "U"),
        ("ML-RANS 5M", cc.solver_meshes(case, "ML-RANS 5M"), "U"),
        ("ML-RANS no clustering", cc.solver_meshes(case, "ML-RANS no clustering"), "U"),
        ("RANS", cc.rans_ref_meshes(case), "U"),
        ("LES", les_paths, les_field),
    ]
    simulations = {}
    for label, paths, field in raw:
        meshes = [pv.read(p) for p in paths]
        mesh = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
        simulations[label] = (interpolate_cell_fields_to_points(mesh, [field]), field)
    return simulations, les_has_k


def plot_comparison_figure(stations, sampled_by_station, labels, simulations_by_label, uref, output_path):
    """All stations of one comparison group in a single vector figure, sized for
    the full text width of the paper (so the fonts print at 7-8 pt)."""
    ncols = (len(stations) + 1) // 2
    style = {
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "lines.linewidth": 0.9,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, ncols, figsize=(6.5, 4.2), sharey=True)
        for ax, station in zip(axes.flat, stations):
            for label in labels:
                _, velocity_field = simulations_by_label[label]
                plot_velocity_deficit(label, sampled_by_station[station][label], velocity_field, uref, ax=ax, marker_size=6)
            ax.set_title(f"x/D = {station:g}")
        for ax in axes.flat[len(stations):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel("z/D")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"$\Delta U / U_\infty$")
        handles, legend_labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, [cc.display_label(label) for label in legend_labels],
                   loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        cc.save_figure(output_path, fig)
        plt.close(fig)


def plot_integral_metric(case, uref, metric_x, metric_results, output_path):
    """Signed integral of (deficit_sim - deficit_LES) over z/D versus downstream distance."""
    plt.figure()
    plt.axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    for label in METRIC_LABELS:
        if label == "RANS":
            plt.plot(metric_x, metric_results[label], label=label, color="black", linestyle="--", marker="^", markersize=4)
        else:
            plt.plot(metric_x, metric_results[label], label=label, color=ML_COLORS[label], linestyle="-", marker="o", markersize=3)
    plt.xlabel("x/D")
    plt.ylabel(r"$\int (\Delta U_{\mathrm{sim}} - \Delta U_{\mathrm{LES}})\, \mathrm{d}(z/D)$")
    plt.title(rf"$U_\infty = {uref:g}$ m/s")
    plt.legend()
    cc.save_figure(output_path)
    plt.close()


def save_integral_metric_csv(metric_x, metric_results, output_path):
    """Values behind the integral-difference plot, one column per model."""
    header = ",".join(["x/D"] + METRIC_LABELS)
    table = np.column_stack([metric_x] + [metric_results[label] for label in METRIC_LABELS])
    np.savetxt(output_path, table, delimiter=",", header=header, comments="", fmt="%.6g")


# wake region where the next turbine of a farm is placed (up to the last station)
FAR_WAKE_START = 4.0


def save_integral_summary_csv(abs_metrics_by_case, output_path):
    """Station-averaged absolute metric per model: near wake, far wake, all stations."""
    rows = ["case,model,near_wake_x/D<4,far_wake_x/D>=4,all_stations"]
    for case, (metric_x, metric_abs) in abs_metrics_by_case.items():
        far = np.asarray(metric_x) >= FAR_WAKE_START
        for label in METRIC_LABELS:
            values = np.asarray(metric_abs[label])
            rows.append(f"{case},{label},{values[~far].mean():.4f},{values[far].mean():.4f},{values.mean():.4f}")
    Path(output_path).write_text("\n".join(rows) + "\n", encoding="utf-8")


def plot_integral_panels(metrics_by_case, output_path, ylabel, zero_line=False, far_wake_band=False):
    """The three inflow cases side by side in one vector figure, sized for the
    full text width of the paper, with a single legend."""
    style = {
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "lines.linewidth": 0.9,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, len(metrics_by_case), figsize=(6.5, 2.5))
        for ax, (tag, (case, (metric_x, metric_results))) in zip(axes, zip("abc", metrics_by_case.items())):
            if zero_line:
                ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
            if far_wake_band:
                ax.axvspan(FAR_WAKE_START, max(metric_x), color="0.9", zorder=0, linewidth=0)
            for label in METRIC_LABELS:
                if label == "RANS":
                    ax.plot(metric_x, metric_results[label], label=label, color="black", linestyle="--", marker="^", markersize=3)
                else:
                    ax.plot(metric_x, metric_results[label], label=label, color=ML_COLORS[label], linestyle="-", marker="o", markersize=2.5)
            ax.set_title(rf"({tag}) $U_\infty = {cc.UREF[case]:g}$ m/s")
            ax.set_xlabel("x/D")
        axes[0].set_ylabel(ylabel)
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="lower center", ncol=len(METRIC_LABELS), frameon=False)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        cc.save_figure(output_path, fig)
        plt.close(fig)


metrics_by_case = {}
abs_metrics_by_case = {}
for case in cc.CASES:
    uref = cc.UREF[case]
    simulations_by_label, les_has_k = build_case_simulations(case)
    case_root = OUTPUT_ROOT / case / "deficit"

    # prepare output dirs
    group_dirs = {}
    for comparison_name, _ in comparison_groups:
        velocity_dir = case_root / comparison_name / "velocity"
        k_dir = case_root / comparison_name / "k"
        velocity_dir.mkdir(parents=True, exist_ok=True)
        if les_has_k:
            k_dir.mkdir(parents=True, exist_ok=True)
        group_dirs[comparison_name] = (velocity_dir, k_dir)

    metric_results = {label: [] for label in METRIC_LABELS}   # signed: net bias
    metric_abs = {label: [] for label in METRIC_LABELS}       # absolute: profile error
    metric_x = []
    sampled_by_station = {}

    for i in x:
        line = pv.Line((i * rotor_diameter, 0, 0), (i * rotor_diameter, 0, 5 * rotor_radius), resolution=1000)
        # sample every mesh on this line once and reuse across groups/metric
        sampled = {label: line.sample(mesh) for label, (mesh, _) in simulations_by_label.items()}
        sampled_by_station[i] = sampled

        for comparison_name, labels in comparison_groups:
            velocity_dir, k_dir = group_dirs[comparison_name]

            plt.figure()
            for label in labels:
                _, velocity_field = simulations_by_label[label]
                plot_velocity_deficit(label, sampled[label], velocity_field, uref)
            plt.xlabel(r"$\Delta U / U_\infty$")
            plt.ylabel("z/D")
            if i == x[-1]:
                plt.legend(fontsize=11, loc="upper right")
            plt.title(f"x/D={i}")
            plt.savefig(velocity_dir / f"velocity_deficit_x_{i}D.png", bbox_inches='tight')
            plt.close()

            if les_has_k:
                plt.figure()
                for label in labels:
                    plot_k_deficit(label, sampled[label])
                plt.xlabel(r"$k$")
                plt.ylabel("z/D")
                if i == 0.25:
                    plt.legend()
                plt.title(f"x/D={i}")
                plt.savefig(k_dir / f"k_deficit_x_{i}D.png", bbox_inches='tight')
                plt.close()

        # integral-difference metric (downstream stations only)
        if i in x_metric:
            les_field = simulations_by_label["LES"][1]
            les_deficit, z_over_d = velocity_deficit_profile(sampled["LES"], les_field, uref)
            metric_x.append(i)
            for label in METRIC_LABELS:
                sim_field = simulations_by_label[label][1]
                sim_deficit, _ = velocity_deficit_profile(sampled[label], sim_field, uref)
                metric_results[label].append(float(np.trapz(sim_deficit - les_deficit, z_over_d)))
                metric_abs[label].append(float(np.trapz(np.abs(sim_deficit - les_deficit), z_over_d)))

    # all ml_models_comparison stations in one vector figure
    plot_comparison_figure(
        x, sampled_by_station, dict(comparison_groups)["ml_models_comparison"], simulations_by_label, uref,
        group_dirs["ml_models_comparison"][0] / "comparison.png",
    )

    # integral-difference metric plot and values for this inflow case
    plot_integral_metric(case, uref, metric_x, metric_results, OUTPUT_ROOT / case / "integral_deficit_difference.png")
    save_integral_metric_csv(metric_x, metric_results, OUTPUT_ROOT / case / "integral_deficit_difference.csv")
    save_integral_metric_csv(metric_x, metric_abs, OUTPUT_ROOT / case / "integral_deficit_error.csv")
    metrics_by_case[case] = (metric_x, metric_results)
    abs_metrics_by_case[case] = (metric_x, metric_abs)

# the three inflow cases in one figure: signed bias (reference) and absolute
# profile error with the far-wake band (the one used in the paper)
plot_integral_panels(metrics_by_case, OUTPUT_ROOT / "integral_deficit_difference.png", r"$I$", zero_line=True)
plot_integral_panels(abs_metrics_by_case, OUTPUT_ROOT / "integral_deficit_error.png", r"$E$", far_wake_band=True)
save_integral_summary_csv(abs_metrics_by_case, OUTPUT_ROOT / "integral_deficit_error_summary.csv")
