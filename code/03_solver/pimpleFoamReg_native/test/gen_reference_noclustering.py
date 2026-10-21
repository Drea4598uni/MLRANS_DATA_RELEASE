#!/usr/bin/env python3
"""Riferimento Python per la pipeline NO-CLUSTERING (predict_corrections_from_cartesian).
Uso: gen_reference_noclustering.py <workdir>"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
NOCL_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "pimpleFoamReg_noclustering"))
MODEL_DIR = os.path.join(NOCL_DIR, "models")

FV = """turbine1
{
    axialFlowTurbineALSourceCoeffs
    {
        origin (0 0 90); axis (-1 0 0); rotorRadius 63.0;
        freeStreamVelocity (8 0 0); tipSpeedRatio 7.55;
    }
}
"""
TRIM = "towerHeight 87.6;\nhubAxDist 5.0191;\nrotorYaw 0.0;\nrotorTilt 0.0872664626;\n"

def main():
    wd = sys.argv[1]
    os.makedirs(os.path.join(wd, "system"), exist_ok=True)
    open(os.path.join(wd, "system", "fvOptions"), "w").write(FV)
    open(os.path.join(wd, "system", "Trim.turbine1"), "w").write(TRIM)
    os.chdir(wd)
    sys.path.insert(0, NOCL_DIR)

    rng = np.random.default_rng(7)
    n = 20000
    x = rng.uniform(-200, 800, n); y = rng.uniform(-200, 200, n); z = rng.uniform(0, 300, n)
    ux = rng.uniform(4, 14, n); uy = rng.uniform(-2, 2, n); uz = rng.uniform(-2, 2, n)
    k = rng.uniform(1e-4, 2, n); nut = rng.uniform(1e-5, 20, n)
    np.savetxt("inputs.txt", np.column_stack([x, y, z, ux, uy, uz, k, nut]),
               fmt="%.17g", header=str(n), comments="")

    from neural_network import NeuralNetworkCorrector
    corr = NeuralNetworkCorrector(
        os.path.join(MODEL_DIR, "best_model_nut.keras"),
        os.path.join(MODEL_DIR, "preprocessor.pkl"),
        os.path.join(MODEL_DIR, "y_scaler_nut.pkl"))
    (nut_c, dnut, kc, dk, u_ax, u_rad, u_tan, ax, rho, theta) = \
        corr.predict_corrections_from_cartesian(ux, uy, uz, x, y, z, k, nut)

    zero = np.zeros(n)
    np.savetxt("expected.txt",
        np.column_stack([u_ax, u_rad, u_tan, zero, zero, ax, rho, theta, nut_c, dnut]),
        fmt="%.17g", header=str(n), comments="")
    print(f"OK n={n}, mode={corr._target_mode}, transform={corr._target_transform}")

if __name__ == "__main__":
    main()
