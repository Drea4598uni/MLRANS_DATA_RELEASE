#!/usr/bin/env python3
"""Confronta expected.txt (pipeline Python) con native_out.txt (C++)."""
import sys

import numpy as np

COLS = [
    ("u_ax", 1e-9, "rel"),
    ("u_rad", 1e-9, "rel"),
    ("u_tan", 1e-9, "rel"),
    ("clusterID", 0.0, "exact"),
    ("centroid_dist", 1e-9, "rel"),
    ("ax", 1e-9, "rel"),
    ("rho", 1e-9, "rel"),
    ("theta", 1e-9, "rel"),
    ("nut_corrected", 2e-4, "abs+rel"),
    ("delta_nut", 2e-4, "abs+rel"),
]


def main():
    ref = np.loadtxt(sys.argv[1], skiprows=1)
    got = np.loadtxt(sys.argv[2], skiprows=1)
    assert ref.shape == got.shape, f"shape mismatch {ref.shape} vs {got.shape}"

    ok = True
    for j, (name, tol, kind) in enumerate(COLS):
        r, g = ref[:, j], got[:, j]
        if kind == "exact":
            n_bad = int(np.sum(r != g))
            status = "OK" if n_bad == 0 else f"FAIL ({n_bad} diversi)"
            ok &= n_bad == 0
            print(f"  {name:15s} {status}")
            continue
        denom = np.maximum(np.abs(r), 1.0 if kind == "abs+rel" else 1e-30)
        err = np.abs(g - r) / denom
        worst = float(err.max())
        status = "OK" if worst <= tol else "FAIL"
        ok &= worst <= tol
        print(f"  {name:15s} max_err={worst:.3e} (tol={tol:g}) {status}")

    if ok:
        print("\n✓ TUTTI I CONFRONTI SUPERATI: pipeline C++ equivalente a Python")
        return 0
    print("\n✗ CONFRONTO FALLITO")
    return 1


if __name__ == "__main__":
    sys.exit(main())
