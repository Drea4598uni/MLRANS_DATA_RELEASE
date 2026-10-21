#!/usr/bin/env python3
"""Disk-averaged wake metrics from the cross-plane samples.

Expected layout (built by the steps in ISTRUZIONI.md):
  cross_planes_export/<case>/<model>/*.csv     case in {8ms, 11ms, 15ms}
OpenFOAM writes, per set, one csv per field type (e.g. disk_x5D_k.csv and
disk_x5D_U.csv, or one file with all fields): every csv whose name starts with
"disk_x<X>D" is read and merged on the point coordinates.

Model folders: the one named "LES" is the reference of its case. Velocity is
taken from UMean (LES) or U (RANS, ML-RANS; their UMean averages from t = 0 and
includes the start-up transient). k is k (RANS, ML-RANS) or, for the LES,
0.5*trace(UPrime2Mean) plus the SGS part if a kMean or
turbulenceProperties:RMean column is present (OpenFOAM symmTensor order:
XX XY XZ YY YZ ZZ, trace = components 0, 3, 5).

Output: disk_metrics.csv with, per case / model / station:
  coverage            fraction of disk points found in the mesh
  REWS, REWS_Uinf     rotor-equivalent wind speed (<Ux^3>_disk)^(1/3)
  P_err_pct           100*(<Ux^3>/<Ux^3>_LES - 1): error on the available power
  P_err_upper/lower/left/right_pct   same on the four half-disks
  TI_rotor_pct        100*sqrt(2/3 <k>_disk)/U_inf
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

D, R, HUB_Z = 126.0, 63.0, 90.0
N_R, N_THETA = 48, 96                     # as in make_sample_dict.py
UINF = {"8ms": 8.0, "11ms": 11.4, "15ms": 15.0}


def read_set(folder, set_name):
    frames = []
    for f in sorted(folder.glob(f"{set_name}*.csv")):
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        frames.append(df.round({"x": 4, "y": 4, "z": 4}))
    if not frames:
        raise FileNotFoundError(f"no csv for {set_name} in {folder}")
    out = frames[0]
    for df in frames[1:]:
        new = [c for c in df.columns if c not in out.columns or c in ("x", "y", "z")]
        out = out.merge(df[new], on=["x", "y", "z"], how="inner")
    return out


VECTOR = ("x", "y", "z")
SYMM_TENSOR = ("xx", "xy", "xz", "yy", "yz", "zz")   # OpenFOAM order


def component(df, base, i):
    """Column of component i of a vector (0-2) or symmTensor (0-5) field, whatever
    the naming used by the csv writer (U_0, U_x, Ux, UPrime2Mean_XX, ...)."""
    suffixes = [str(i)]
    for names in (VECTOR, SYMM_TENSOR):
        if i < len(names):
            suffixes += [names[i], names[i].upper()]
    for sfx in suffixes:
        for name in (f"{base}_{sfx}", f"{base}{sfx}", f"{base}:{sfx}"):
            if name in df.columns:
                return df[name].to_numpy(float)
    raise KeyError(f"{base} component {i} not found in {list(df.columns)}")


def velocity_x(df, is_les):
    return component(df, "UMean" if is_les else "U", 0)


def tke(df, is_les):
    if not is_les:
        return df["k"].to_numpy(float)
    trace = sum(component(df, "UPrime2Mean", i) for i in (0, 3, 5))
    k = 0.5 * trace
    if "kMean" in df.columns:
        k = k + df["kMean"].to_numpy(float)
    elif any(c.startswith("turbulenceProperties:RMean") for c in df.columns):
        k = k + 0.5 * sum(component(df, "turbulenceProperties:RMean", i) for i in (0, 3, 5))
    return k


def disk_weights(df):
    """Polar midpoint weights r*dr*dtheta, normalised on the full disk."""
    r = np.hypot(df["y"].to_numpy(float), df["z"].to_numpy(float) - HUB_Z)
    w = r * (R / N_R) * (2 * np.pi / N_THETA) / (np.pi * R**2)
    return w


def metrics(df, is_les, uinf):
    w = disk_weights(df)
    ux, k = velocity_x(df, is_les), tke(df, is_les)
    y, z = df["y"].to_numpy(float), df["z"].to_numpy(float)
    res = {"coverage": w.sum()}
    halves = {"all": np.ones_like(w, bool), "upper": z > HUB_Z, "lower": z < HUB_Z, "left": y > 0, "right": y < 0}
    for h, sel in halves.items():
        res[f"u3_{h}"] = np.sum(w[sel] * ux[sel] ** 3) / np.sum(w[sel])
    res["REWS"] = res["u3_all"] ** (1 / 3)
    res["REWS_Uinf"] = res["REWS"] / uinf
    res["TI_rotor_pct"] = 100 * np.sqrt(2 / 3 * np.sum(w * k) / np.sum(w)) / uinf
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="cross_planes_export")
    ap.add_argument("--out", default="disk_metrics.csv")
    args = ap.parse_args()
    rows = []
    for case_dir in sorted(p for p in Path(args.root).iterdir() if p.is_dir() and p.name in UINF):
        uinf = UINF[case_dir.name]
        models = {m.name: m for m in case_dir.iterdir() if m.is_dir()}
        if "LES" not in models:
            print(f"!! {case_dir.name}: no LES folder, skipped")
            continue
        stations = sorted({int(re.match(r"disk_x(\d+)D", f.name).group(1))
                           for f in models["LES"].glob("disk_x*D*.csv")})
        ref = {s: metrics(read_set(models["LES"], f"disk_x{s}D"), True, uinf) for s in stations}
        for name, folder in sorted(models.items()):
            for s in stations:
                m = metrics(read_set(folder, f"disk_x{s}D"), name == "LES", uinf)
                row = {"case": case_dir.name, "model": name, "x_D": s, "coverage": round(m["coverage"], 4),
                       "REWS": m["REWS"], "REWS_Uinf": m["REWS_Uinf"], "TI_rotor_pct": m["TI_rotor_pct"]}
                for h in ("all", "upper", "lower", "left", "right"):
                    key = "P_err_pct" if h == "all" else f"P_err_{h}_pct"
                    row[key] = 100 * (m[f"u3_{h}"] / ref[s][f"u3_{h}"] - 1)
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table[["case", "model", "x_D", "coverage", "REWS_Uinf", "P_err_pct", "TI_rotor_pct"]].round(3))
    print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
