#!/usr/bin/env python3
"""Write the OpenFOAM function-object dictionary that samples the cross-stream
planes x = const behind the rotor (NREL 5MW, D = 126 m, hub at (0, 0, 90) m).

For each station x/D it defines two point clouds:
  disk_x<X>D   polar grid on the rotor disk (r <= R), used for the disk averages
               (midpoint rule in r and theta: weights r*dr*dtheta are rebuilt
               from the coordinates by disk_metrics.py)
  plane_x<X>D  Cartesian grid y in [-1.5D, 1.5D], z in [1 m, 2D], 2 m spacing,
               used only for maps / checks

Usage (one dictionary per field set):
  python3 make_sample_dict.py --fields "U k" --out crossPlanesDict          # RANS, ML-RANS
  python3 make_sample_dict.py --fields "UMean UPrime2Mean" --out crossPlanesDict_LES
Add SGS fields of the LES to --fields if they exist in its latest time
directory (e.g. kMean, turbulenceProperties:RMean): see ISTRUZIONI.md.
"""
import argparse

import numpy as np

D, R = 126.0, 63.0
HUB = np.array([0.0, 0.0, 90.0])
STATIONS = [2, 3, 4, 5, 6, 7]
N_R, N_THETA = 48, 96
PLANE_STEP = 2.0


def disk_points(x):
    dr = R / N_R
    r = (np.arange(N_R) + 0.5) * dr
    theta = (np.arange(N_THETA) + 0.5) * 2 * np.pi / N_THETA
    rr, tt = np.meshgrid(r, theta, indexing="ij")
    y = rr * np.cos(tt)
    z = HUB[2] + rr * np.sin(tt)
    return np.column_stack([np.full(y.size, x), y.ravel(), z.ravel()])


def plane_points(x):
    y = np.arange(-1.5 * D, 1.5 * D + 1e-9, PLANE_STEP)
    z = np.arange(1.0, 2 * D + 1e-9, PLANE_STEP)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def cloud(name, pts):
    body = "\n".join(f"                ({p[0]:.4f} {p[1]:.4f} {p[2]:.4f})" for p in pts)
    return (f"        {name}\n        {{\n            type    cloud;\n            axis    xyz;\n"
            f"            points\n            (\n{body}\n            );\n        }}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="U k", help="space-separated field names")
    ap.add_argument("--out", default="crossPlanesDict")
    ap.add_argument("--name", default="crossPlanes", help="function-object name = postProcessing sub-folder")
    args = ap.parse_args()
    fields = " ".join(f'"{f}"' if ":" in f else f for f in args.fields.split())
    sets = ""
    for s in STATIONS:
        x = s * D + HUB[0]
        sets += cloud(f"disk_x{s}D", disk_points(x))
        sets += cloud(f"plane_x{s}D", plane_points(x))
    text = f"""/*--------------------------------*- C++ -*----------------------------------*\\
  Cross-stream planes behind the rotor, written by make_sample_dict.py
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {args.out};
}}

{args.name}
{{
    type                sets;
    libs                (sampling);
    interpolationScheme cellPoint;
    setFormat           csv;
    fields              ({fields});
    sets
    {{
{sets}    }}
}}
"""
    with open(args.out, "w") as fh:
        fh.write(text)
    n = len(STATIONS) * (N_R * N_THETA + len(plane_points(0.0)))
    print(f"wrote {args.out}: {len(STATIONS)} stations, {n} points, fields ({fields})")


if __name__ == "__main__":
    main()
