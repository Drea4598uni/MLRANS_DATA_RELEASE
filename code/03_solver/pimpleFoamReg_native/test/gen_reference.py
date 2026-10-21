#!/usr/bin/env python3
"""
Genera input sintetici e output di riferimento usando la pipeline Python
ORIGINALE di pimpleFoamReg_pt2 (clustering.py + neural_network.py), replicando
esattamente la sequenza di chiamate del solver C++:

    perform_clustering_with_distances(ux,uy,uz,x,y,z,k,nut)
    predict_corrections(u_ax,u_tan,u_rad,k,nut,centroid_dist,ids,ax,z)

Uso:
    python3.10 gen_reference.py <workdir>

In <workdir> scrive: system/fvOptions, system/Trim.turbine1,
inputs.txt (x y z ux uy uz k nut) e expected.txt.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PT2_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "pimpleFoamReg_pt2"))
MODEL_DIR = os.path.normpath(os.path.join(HERE, "..", "models"))

FV_OPTIONS = """/*--------------------------------*- C++ -*----------------------------------*\\
| Test case for nativeML validation                                           |
\\*---------------------------------------------------------------------------*/
turbine1
{
    type            axialFlowTurbineALSource;
    active          on;

    axialFlowTurbineALSourceCoeffs
    {
        fieldNames          (U);
        origin              (0 0 90);
        axis                (-1 0 0);
        rotorRadius         63.0;
        freeStreamVelocity  (8 0 0);
        tipSpeedRatio       7.55;
    }
}
"""

TRIM = """towerHeight 87.6;
hubAxDist 5.0191;
rotorYaw 0.0;
rotorTilt 0.0872664626;
"""


def main():
    workdir = sys.argv[1]
    os.makedirs(os.path.join(workdir, "system"), exist_ok=True)
    with open(os.path.join(workdir, "system", "fvOptions"), "w") as f:
        f.write(FV_OPTIONS)
    with open(os.path.join(workdir, "system", "Trim.turbine1"), "w") as f:
        f.write(TRIM)

    # La pipeline legge system/* dal cwd, come nel solver
    os.chdir(workdir)
    sys.path.insert(0, PT2_DIR)

    rng = np.random.default_rng(42)
    n = 20000
    x = rng.uniform(-200.0, 800.0, n)
    y = rng.uniform(-200.0, 200.0, n)
    z = rng.uniform(0.0, 300.0, n)
    ux = rng.uniform(4.0, 14.0, n)
    uy = rng.uniform(-2.0, 2.0, n)
    uz = rng.uniform(-2.0, 2.0, n)
    k = rng.uniform(1e-4, 2.0, n)
    nut = rng.uniform(1e-5, 20.0, n)

    np.savetxt(
        "inputs.txt",
        np.column_stack([x, y, z, ux, uy, uz, k, nut]),
        fmt="%.17g",
        header=str(n),
        comments="",
    )

    from clustering import ClusteringProcessor
    from neural_network import NeuralNetworkCorrector

    proc = ClusteringProcessor(
        os.path.join(MODEL_DIR, "km"), os.path.join(MODEL_DIR, "km-scaler")
    )
    corr = NeuralNetworkCorrector(
        os.path.join(MODEL_DIR, "best_model_nut.keras"),
        os.path.join(MODEL_DIR, "preprocessor.pkl"),
        os.path.join(MODEL_DIR, "y_scaler_nut.pkl"),
    )

    (
        u_ax,
        u_rad,
        u_tan,
        ids,
        cdist,
        ax,
        rho,
        theta,
    ) = proc.perform_clustering_with_distances(ux, uy, uz, x, y, z, k, nut)

    nut_corr, delta_nut, _, _ = corr.predict_corrections(
        u_ax, u_tan, u_rad, k, nut, cdist, ids, ax, z
    )

    np.savetxt(
        "expected.txt",
        np.column_stack(
            [u_ax, u_rad, u_tan, ids.astype(float), cdist, ax, rho, theta,
             nut_corr, delta_nut]
        ),
        fmt="%.17g",
        header=str(n),
        comments="",
    )
    print(f"\n✓ Riferimento generato in {workdir} (n={n})")
    print(f"  target_mode={corr._target_mode}, transform={corr._target_transform}")


if __name__ == "__main__":
    main()
