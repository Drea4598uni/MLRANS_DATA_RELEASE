#!/usr/bin/env python3
"""
Coerenza della pipeline nativa con il TRAINING dei modelli 1T.

I modelli in models/ (KMeans 7 cluster su [Ux Ut Ur]) sono stati addestrati
sulle slice del RANS 1T 11ms_simpleFoam_corretto (sim_clustering/data/
yNormal.vtk, punti con Ux >= min dello scaler). Lo script:
  1. ricalcola il frame cilindrico con le formule del repo di training
     (utils.sim_directory/rotation/points_CART2CYL/vector_CART2CYL, copiate
     qui: indipendenti da pt2 e da nativeML.H) e controlla che le feature
     riproducano ESATTAMENTE data_min_/data_max_ del km-scaler;
  2. esegue testNative sugli stessi punti con fvOptions/Trim del caso e
     confronta Uax/Ur/Ut/ax/theta e clusterID (sklearn + riordino Ux desc).

Uso (venv acTestEnv, da test/ dopo aver compilato testNative):
    python3.10 check_training_1T.py <workdir> [<params .dat>]
"""
import os
import shutil
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pyvista as pv

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
CASE = "/mnt/NAS/andrea/from_frank/OF/5MW_1T/5MW_RANS/terrain/11ms_simpleFoam_corretto"
DATA = os.path.join(CASE, "sim_clustering/data/yNormal.vtk")
MODELS = os.path.join(HERE, "..", "models")


def training_frame(case_dir, turbine="turbine1"):
    """Replica utils.sim_directory + rotation del repo sim_clustering."""
    cur, tur = None, {}
    for line in open(os.path.join(case_dir, "system/fvOptions")):
        st = line.strip()
        if st.startswith(("//", "/*", "*")):
            continue
        if st.startswith("turbine"):
            cur = st.split()[0]
            tur[cur] = {}
        if cur and "origin" in line:
            tur[cur]["origin"] = [float(v) for v in line.split("(")[1].split(")")[0].replace(";", "").split()]
        elif cur and "axis" in line:
            tur[cur]["axis"] = [float(v) for v in line.split("(")[1].split(")")[0].replace(";", "").split()]
    z_tower = hub = yaw = tilt = 0.0
    for line in open(os.path.join(case_dir, f"system/Trim.{turbine}")):
        if "towerHeight " in line:
            z_tower = float(line.split()[1].replace(";", ""))
        elif "hubAxDist " in line:
            hub = -float(line.split()[1].replace(";", ""))
        elif "rotorYaw " in line:
            yaw = float(line.split()[1].replace(";", ""))
        elif "rotorTilt " in line:
            tilt = float(line.split()[1].replace(";", ""))
    Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rt = np.array([[np.cos(tilt), 0, np.sin(tilt)], [0, 1, 0], [-np.sin(tilt), 0, np.cos(tilt)]])
    R = Ry @ Rt
    axis = -R @ np.array(tur[turbine]["axis"])
    ref = R @ np.array([0.0, -1.0, 0.0])
    o = tur[turbine]["origin"]
    return axis, ref, np.array([o[0] + hub, o[1], o[2] + z_tower])


def training_cyl(P, U, axis, ref, shift):
    """utils.points_CART2CYL + vector_CART2CYL (cyl_abs False)."""
    q = P - shift
    proj = q - np.outer(q @ axis, axis)
    th = np.arctan2(np.cross(ref, proj) @ axis, proj @ ref)
    ur = np.cos(th)*U[:, 1] + np.sin(th)*U[:, 2]
    ut = -np.sin(th)*U[:, 1] + np.cos(th)*U[:, 2]
    return U @ axis, ur, ut, q @ axis, th


def main():
    workdir = os.path.abspath(sys.argv[1])
    params = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(MODELS, "native_ml.dat")
    os.makedirs(os.path.join(workdir, "system"), exist_ok=True)
    for f in os.listdir(os.path.join(CASE, "system")):
        if f == "fvOptions" or f.startswith("Trim."):
            shutil.copy2(os.path.join(CASE, "system", f), os.path.join(workdir, "system", f))

    sc = joblib.load(os.path.join(MODELS, "km-scaler"))
    km = joblib.load(os.path.join(MODELS, "km"))
    feats = list(sc.feature_names_in_)

    m = pv.read(DATA)
    P = np.asarray(m.points, np.float64)
    U = np.asarray(m.point_data["U"], np.float64)
    k = np.asarray(m.point_data["k"], np.float64)
    nut = np.asarray(m.point_data["nut"], np.float64)
    sel = U[:, 0] >= sc.data_min_[feats.index("Ux")]
    P, U, k, nut = P[sel], U[sel], k[sel], nut[sel]

    axis, ref, shift = training_frame(CASE)
    uax, ur, ut, ax, th = training_cyl(P, U, axis, ref, shift)
    X = pd.DataFrame({"Ux": uax, "Ut": ut, "Ur": ur})[feats]
    dmin = np.abs(X.min().values - sc.data_min_).max()
    dmax = np.abs(X.max().values - sc.data_max_).max()
    print(f"  scaler di training riprodotto: |min diff|={dmin:.2e} |max diff|={dmax:.2e} (n={len(P)})")
    ok = dmin < 1e-9 and dmax < 1e-9

    raw = km.predict(sc.transform(X))
    order = np.argsort(km.cluster_centers_[:, feats.index("Ux")], kind="stable")[::-1]
    remap = np.empty(km.n_clusters, dtype=int)
    remap[order] = np.arange(1, km.n_clusters + 1)
    ids = remap[raw]

    n = len(P)
    inp = os.path.join(workdir, "inputs.txt")
    out = os.path.join(workdir, "native_out.txt")
    np.savetxt(inp, np.column_stack([P, U, k, nut]), fmt="%.17g", header=str(n), comments="")
    subprocess.run([os.path.join(HERE, "testNative"), params, workdir, inp, out], check=True)
    got = np.loadtxt(out, skiprows=1)

    for name, j, r in [("Uax", 0, uax), ("Ur", 1, ur), ("Ut", 2, ut), ("ax", 5, ax), ("theta", 7, th)]:
        d = np.angle(np.exp(1j*(got[:, j] - r))) if name == "theta" else got[:, j] - r
        err = float(np.max(np.abs(d)))
        print(f"  {name:6s} max|diff| = {err:.3e}")
        ok &= err < 1e-6
    same = float(np.mean(got[:, 3] == ids))
    print(f"  ID     identici: {100*same:.4f}%  ({int(np.sum(got[:, 3] != ids))} diversi)")
    ok &= same > 0.9999
    print("\n✓ pipeline nativa coerente col training 1T" if ok else "\n✗ INCOERENTE col training 1T")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
