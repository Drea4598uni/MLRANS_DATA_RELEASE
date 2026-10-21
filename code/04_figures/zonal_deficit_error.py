"""Error of the velocity-deficit profiles with respect to the LES, by region of
the vertical profile (table tab:zonal_error of the paper).

Regions of the vertical line y = 0 behind the rotor:
  ground layer   0    <= z/D < 0.214   (below the lower blade tip)
  rotor disk     0.214 <= z/D <= 1.214
  upper edge     1.214 <  z/D <= 1.6   (wake edge above the upper tip)
Stations (as in the deficit figures) in three groups: near wake x/D = 0.5-3,
x/D = 4-5 and far wake x/D = 6-7. Deficit dU/U_inf = 1 - |U|/U_inf, as in Figs. 12, 14, 15
(RANS and ML-RANS: U at the last time; LES: UMean).
For each region and group of stations: mean absolute error (the table of the
paper) and mean signed error (> 0: deficit overestimated), in % of U_inf.

Run from the package root:
    python code/04_figures/zonal_deficit_error.py
Writes figures_output/ml_results/zonal_deficit_error.csv and prints the LaTeX rows.
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc  # noqa: E402

GROUPS = {"near": [0.5, 1, 2, 3], "mid": [4, 5], "far": [6, 7]}
BANDS = {"ground": (0.0, 0.214), "disk": (0.214, 1.214), "edge": (1.214, 1.6)}
MODELS = ["RANS", "ML-RANS 15M", "ML-RANS 10M", "ML-RANS 5M", "ML-RANS no clustering"]
OUTPUT = Path("figures_output/ml_results/zonal_deficit_error.csv")
z_d = np.linspace(0.0025, 1.6, 640)


def meshes(case):
    les_paths, les_field, _ = cc.les_ref(case)
    out = {"LES": (pv.read(les_paths[0]), les_field), "RANS": (pv.read(cc.rans_ref_meshes(case)[0]), "U")}
    for label in MODELS[1:]:
        out[label] = (pv.read(cc.solver_meshes(case, label)[0]).cell_data_to_point_data(), "U")
    return out


def deficit(mesh, field, x_d, uinf):
    pts = np.column_stack([np.full_like(z_d, x_d * cc.ROTOR_DIAMETER), np.zeros_like(z_d), z_d * cc.ROTOR_DIAMETER])
    sampled = pv.PolyData(pts).sample(mesh)
    valid = np.asarray(sampled["vtkValidPointMask"]).astype(bool)
    speed = np.linalg.norm(np.asarray(sampled[field]), axis=1)
    return np.interp(z_d, z_d[valid], 1 - speed[valid] / uinf)


def main():
    rows, latex = ["case,model,group,band,mean_abs_pct,mean_signed_pct"], []
    for case in cc.CASES:
        uinf = cc.UREF[case]
        data = meshes(case)
        les = {x: deficit(*data["LES"], x, uinf) for g in GROUPS.values() for x in g}
        for model in MODELS:
            cells = []
            for group_name, group in GROUPS.items():
                err = np.array([deficit(*data[model], x, uinf) - les[x] for x in group])
                for band, (lo, hi) in BANDS.items():
                    sel = (z_d >= lo) & (z_d <= hi) if band == "disk" else (z_d >= lo) & (z_d < hi) if band == "ground" else (z_d > lo) & (z_d <= hi)
                    mean_abs, mean_signed = 100 * np.abs(err[:, sel]).mean(), 100 * err[:, sel].mean()
                    rows.append(f"{case},{model},{group_name},{band},{mean_abs:.2f},{mean_signed:.2f}")
                    cells.append(f"{mean_abs:.1f} (${mean_signed:+.1f}$)")
            label = "RANS 30.8M" if model == "RANS" else cc.display_label(model)
            latex.append(f"{cc.UREF[case]:g} & {label} & " + " & ".join(cells) + r" \\")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(latex))


if __name__ == "__main__":
    main()
