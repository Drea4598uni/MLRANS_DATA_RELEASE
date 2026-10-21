#!/usr/bin/env python3
"""
Confronta la geometria (parsing fvOptions/Trim) tra la pipeline Python
originale e nativeML::Geometry (geomCheck) su una lista di case dir reali.
Sola lettura. Uso:
    python3.10 geom_compare.py <case_dir> [<case_dir> ...]
"""
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PT2_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "pimpleFoamReg_pt2"))
GEOMCHECK = os.path.join(HERE, "geomCheck")

sys.path.insert(0, PT2_DIR)


def python_geometry(case_dir):
    """Replica l'init geometria di ClusteringProcessor senza caricare i modelli."""
    from clustering import ClusteringProcessor

    proc = ClusteringProcessor.__new__(ClusteringProcessor)
    proc._cyl_abs = False
    proc._turbine_name = None
    proc._turbines = {}
    proc._origin = np.array([0.0, 0.0, 0.0])
    proc._axis_input = np.array([-1.0, 0.0, 0.0])
    proc._ref_axis_base = np.array([0.0, -1.0, 0.0])
    proc._hub_ax_dist = 0.0
    proc._z_tower = 0.0
    proc._rotor_yaw = 0.0
    proc._rotor_tilt = 0.0
    proc._rotor_radius = 63.0
    proc._load_case_config(case_dir)
    proc._setup_rotated_frame()
    return {
        "turbine": proc._turbine_name or "-",
        "rotor_radius": float(proc._rotor_radius),
        "axis_dir": proc._axis_dir,
        "ref_axis": proc._ref_axis,
        "point_shift": proc._point_shift,
    }


def native_geometry(case_dir):
    out = subprocess.run(
        [GEOMCHECK, case_dir], capture_output=True, text=True, check=True
    ).stdout
    vals = {}
    for line in out.strip().splitlines():
        toks = line.split()
        vals[toks[0]] = toks[1] if len(toks) == 2 else np.array(
            [float(v) for v in toks[1:]]
        )
    return vals


def main():
    all_ok = True
    for case in sys.argv[1:]:
        py = python_geometry(case)
        try:
            nat = native_geometry(case)
        except subprocess.CalledProcessError as e:
            print(f"FAIL {case}: geomCheck errore: {e.stderr}")
            all_ok = False
            continue

        errs = []
        if str(py["turbine"]) != str(nat["turbine"]):
            errs.append(f"turbine {py['turbine']} != {nat['turbine']}")
        if abs(py["rotor_radius"] - float(nat["rotor_radius"])) > 1e-12:
            errs.append("rotor_radius")
        for key in ("axis_dir", "ref_axis", "point_shift"):
            if np.max(np.abs(py[key] - nat[key])) > 1e-12:
                errs.append(f"{key}: py={py[key]} nat={nat[key]}")

        if errs:
            all_ok = False
            print(f"FAIL {case}:")
            for e in errs:
                print(f"    {e}")
        else:
            print(
                f"OK   {os.path.basename(case):45s} turbine={py['turbine']} "
                f"axis={py['axis_dir']} shift={py['point_shift']}"
            )

    print("\n" + ("✓ Geometria identica su tutti i casi" if all_ok else "✗ DIFFERENZE"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
