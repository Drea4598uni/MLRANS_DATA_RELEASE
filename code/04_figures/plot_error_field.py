import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc

OUTPUT_ROOT = Path("figures_output/ml_results")

VARIABLES = ("U", "k")
COLOR_PERCENTILES = (5, 95)
ROTOR_DIAMETER = 126.0
ROTOR_CENTER = np.array([0.0, 0.0, 90.0]) / ROTOR_DIAMETER
ROTOR_RADIUS = 63.0 / ROTOR_DIAMETER
ROTOR_LINE_RADIUS = 2.5 / ROTOR_DIAMETER

rotor_line_xz = pv.Line(
    ROTOR_CENTER + np.array([0.0, 0.0, -ROTOR_RADIUS]),
    ROTOR_CENTER + np.array([0.0, 0.0, ROTOR_RADIUS]),
).tube(radius=ROTOR_LINE_RADIUS)
rotor_hub = pv.Sphere(radius=4.0 / ROTOR_DIAMETER, center=ROTOR_CENTER)

bounds_args_xz = {
    "xtitle": "",
    "ytitle": "",
    "ztitle": "",
    "bounds": (-5, 15, 0, 0, 0, 3),
    "fmt": "%.1fD",
    "font_size": 12 * cc.SCREENSHOT_SCALE,  # cube-axes labels are sized in pixels
    "font_family": "times",
    "location": "outer",
    "grid": None,
    "n_xlabels": 5,
    "n_zlabels": 4,
}

scalar_bar_args = {
    "title_font_size": 22,
    "label_font_size": 18,
    "font_family": "times",
    "position_x": 0.2,
    "position_y": 0.17,
    "fmt": "%.1f",
}


def read_case(mesh_paths, velocity_field):
    meshes = [pv.read(path) for path in mesh_paths]
    mesh = pv.merge(meshes) if len(meshes) > 1 else meshes[0]
    cc.scale_to_diameters(mesh)
    return {
        "mesh": cell_data_to_point_data(mesh),
        "velocity_field": velocity_field,
    }


def cell_data_to_point_data(mesh):
    if mesh.cell_data:
        return mesh.cell_data_to_point_data()
    return mesh


def safe_name(label):
    return label.lower().replace(" ", "_").replace("-", "_")


def field_values(mesh, velocity_field, variable):
    if variable == "U":
        velocity = np.asarray(mesh[velocity_field])
        return np.linalg.norm(velocity, axis=1)
    return np.asarray(mesh[variable])


def sampled_reference_values(target_mesh, reference_case, variable):
    sampled = target_mesh.sample(reference_case["mesh"])
    valid_mask = np.asarray(sampled["vtkValidPointMask"], dtype=bool)
    values = field_values(sampled, reference_case["velocity_field"], variable)
    values = values.astype(float)
    values[~valid_mask] = np.nan
    return values


def add_error_field(target_case, reference_case, variable):
    mesh = target_case["mesh"].copy()
    target_values = field_values(mesh, target_case["velocity_field"], variable).astype(float)
    reference_values = sampled_reference_values(mesh, reference_case, variable)
    error_name = f"error_{variable}_percent"
    denominator = np.abs(reference_values)
    error = np.full_like(target_values, np.nan, dtype=float)
    valid = denominator > np.finfo(float).eps
    error[valid] = 100.0 * np.abs(target_values[valid] - reference_values[valid]) / denominator[valid]
    mesh[error_name] = error
    return mesh, error_name


def crop_to_low_z(mesh, z_max=3.0):
    return mesh.clip(normal="z", origin=(0.0, 0.0, z_max), invert=True)


def percentile_color_limits(mesh, scalars, percentiles=COLOR_PERCENTILES):
    values = np.asarray(mesh[scalars])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None

    lower, upper = np.percentile(values, percentiles)
    if np.isclose(lower, upper):
        return None
    return float(lower), float(upper)


def add_xz_axis_titles(plotter):
    plotter.add_text("x/D", position=cc.px((900, 425)), font_size=11, font="times", color="black")
    plotter.add_text("z/D", position=cc.px((250, 585)), font_size=11, font="times", color="black", orientation=90)


def plot_error_mesh(mesh, error_name, output_path, title):
    cropped_mesh = crop_to_low_z(mesh)
    clim = percentile_color_limits(cropped_mesh, error_name)
    plotter = cc.hires_plotter((1800, 1200))
    scalar_args = scalar_bar_args.copy()
    scalar_args["title"] = "Error [%]"
    plotter.add_mesh(
        cropped_mesh,
        scalars=error_name,
        clim=[0,30],
        cmap="turbo",
        nan_color="lightgray",
        show_scalar_bar=True,
        scalar_bar_args=scalar_args,
    )
    plotter.add_mesh(rotor_line_xz, color="black")
    plotter.add_mesh(rotor_hub, color="black")
    add_xz_axis_titles(plotter)
    plotter.add_text(title, position="upper_left", font_size=12, font="times", color="black")
    cc.show_xz_bounds(plotter, bounds_args_xz)
    plotter.view_xz()
    plotter.enable_parallel_projection()
    cc.hires_screenshot(plotter, output_path)
    plotter.close()


comparisons = [
    ("ML-RANS 15M", "LES"),
    ("ML-RANS 10M", "LES"),
    ("ML-RANS 5M", "LES"),
    ("RANS", "LES"),
    ("RANS", "ML-RANS 15M"),
    ("RANS", "ML-RANS 10M"),
    ("RANS", "ML-RANS 5M"),
]


def build_cases(case):
    les_paths, les_field, les_has_k = cc.les_ref(case)
    cases = {
        "ML-RANS 15M": read_case(cc.solver_meshes(case, "ML-RANS 15M"), "U"),
        "ML-RANS 10M": read_case(cc.solver_meshes(case, "ML-RANS 10M"), "U"),
        "ML-RANS 5M": read_case(cc.solver_meshes(case, "ML-RANS 5M"), "U"),
        "RANS": read_case(cc.rans_ref_meshes(case), "U"),
        "LES": read_case(les_paths, les_field),
    }
    return cases, les_has_k


import xz_maps


def paper_error_figure(cases, output_path):
    """Fig. 13: |U| error of ML-RANS 4.2M (a) and of the standard RANS (b) with
    respect to the LES, stacked at full text width with one colour scale."""
    panels = []
    for (target_label, title), tag in zip(
        (("ML-RANS 5M", "ML-RANS 4.2M"), ("RANS", "RANS 30.8M")), "ab"
    ):
        mesh, error_name = add_error_field(cases[target_label], cases["LES"], "U")
        mesh.points = mesh.points * ROTOR_DIAMETER  # back to metres for the resampling
        panels.append({
            "values": xz_maps.resample(mesh, error_name, (-5, 15), (0, 2.5)),
            "title": f"{title} vs LES",
            "tag": tag,
        })
    xz_maps.map_stack(
        panels, output_path,
        shared_colorbar={"cmap": "magma", "clim": (0.0, 30.0), "extend": "max",
                         "label": r"$|\vec{U}|$ error [%]"},
    )


for case in cc.CASES:
    cases, les_has_k = build_cases(case)
    case_root = OUTPUT_ROOT / case / "error_fields"
    if case == cc.PRINCIPAL:
        paper_error_figure(cases, case_root / "error_U_paper.pdf")

    for target_label, reference_label in comparisons:
        target_case = cases[target_label]
        reference_case = cases[reference_label]
        comparison_dir = case_root / f"{safe_name(target_label)}_vs_{safe_name(reference_label)}"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        for variable in VARIABLES:
            # the LES reference at 15 m/s has no k field
            if variable == "k" and reference_label == "LES" and not les_has_k:
                continue
            mesh, error_name = add_error_field(target_case, reference_case, variable)
            output_path = comparison_dir / f"error_{variable}.png"
            title = f"{target_label} vs {reference_label} - {variable} error [%]"
            plot_error_mesh(mesh, error_name, output_path, title)
