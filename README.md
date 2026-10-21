# ML-RANS: code for the CFD-ML eddy-viscosity closure of HAWT wakes

Code supporting the article

> A. Carloni, G. Delibra, F. De Girolamo, L. Tieghi, A. Corsini, V. F. Barnabei,
> *A Machine-Learning-Defined Eddy-Viscosity Closure for RANS Simulations of
> Horizontal-Axis Wind Turbine Wakes*, International Journal of Thermofluids
> (submitted).

A K-means clustering of the flow and a residual Multi-Layer Perceptron (MLP)
are trained on a single LES of the NREL-5MW turbine (actuator line, turbulent
atmospheric boundary layer, hub-height inflow 11.4 m/s) to predict the
LES-equivalent eddy viscosity. The trained models are embedded in a modified
OpenFOAM PIMPLE solver (*ML-RANS*) and run, without retraining, on three grids
and at two inflow velocities never seen in training (8 and 15 m/s).

This repository contains the ML-RANS OpenFOAM solver, the scripts used to
assemble the training data and to train the networks, and the scripts that
produce every data-driven figure and table of the paper. The data are published
on Zenodo: record “ML-RANS: data for the CFD-ML eddy-viscosity closure of horizontal-axis wind turbine wakes”.

**If you use this code, please cite the article above** ("Cite this
repository" on GitHub gives the reference).

Contact: Andrea Carloni, Sapienza University of Rome — andrea.carloni@uniroma1.it

---

## Contents

```
code/
  01_les_postprocessing/   LES slice merging, resolved/SGS k, LES -> RANS interpolation
  02_training/             clustering-informed and plain MLP training scripts
  03_solver/               ML-RANS OpenFOAM solver (native C++ and Python reference)
  04_figures/              scripts that produce the figures and tables of the paper
  05_cross_planes/         sampling dictionary generator and disk metrics (OpenFOAM side)
requirements.txt           Python environment for the figures (code/04)
requirements-training.txt  Python environment for training and model export (code/02, code/03)
LICENSE                    BSD 3-Clause License (all files except code/03_solver)
code/03_solver/LICENSE     GNU General Public License v3.0 (ML-RANS solver)
CITATION.cff               citation metadata
```

## Data

Download the dataset from Zenodo (record “ML-RANS: data for the CFD-ML eddy-viscosity closure of horizontal-axis wind turbine wakes”) and copy its `data/` folder into the
root of this repository, so that the scripts find `data/01_les_reference`,
`data/02_rans_reference`, and so on. The README of the dataset describes every
file. Folder and file names use `8ms`, `11ms` and `15ms` for the inflow
velocities 8, 11.4 (training case) and 15 m/s, and `5ml`, `10ml`, `15ml` for
the 4.2M, 9.1M and 15.9M grids of the ML-RANS runs.

---

## Reproducing the results

### Figures

With the Python environment of `requirements.txt` (Python 3.12):

```bash
pip install -r requirements.txt
python code/04_figures/run_all.py
```

The scripts can be launched from any directory. Outputs are written to
`figures_output/`, and the panels used in the paper are collected in
`figures_output/paper/` under their figure number and panel letter
(`fig9.pdf` is Fig. 9; Figs. 12, 14, 15 and 18 are assembled in the article
from the two parts `a` and `b`). Figures 3 and 7–18 and Table 4 are
generated; Figs. 1, 2, 4, 5 and 6 are schematics or mesh renderings.
Individual scripts can also be run on their own:

| Script | Figures / tables |
|---|---|
| `methodology_fields_fig.py` | 3 |
| `clustering_fig.py` | 7, 8, 9 |
| `NN.py` | 11 |
| `nn_training_figure.py` | 10 (and the metrics of Table 3) |
| `plot_velocity_deficit.py` | 12a, 14a, 15a |
| `plot_fields.py` | 12b, 14b, 15b |
| `plot_error_field.py` | 13 |
| `zonal_deficit_error.py` | Table 4 |
| `plot_k_profiles.py` | 16 |
| `plot_cross_planes.py` | 17 |
| `forces_cp_ct.py` | 18 |

The field maps are drawn by `xz_maps.py`, which resamples the slices on a
regular grid; `case_config.py` holds the shared paths and labels.

### Neural-network training

With the Python 3.10 environment of `requirements-training.txt`:

```bash
pip install -r requirements-training.txt
python code/02_training/train_ml_rans_clustering.py
python code/02_training/train_ml_rans_no_clustering.py
```

The scripts read `data/03_training_data` and `data/04_clustering` and write to
`retrained/` (override with the `OUTPUT_ROOT` environment variable; if the
output folder already contains `best_model_nut.keras`, training is skipped and
only the predictions are recomputed). The training recipe of the paper is the
default: Huber loss (δ = 1), Adam (1.5e-4), at most 220 epochs, early stopping
(patience 20), axial sample weight w = 1 + max(x/D, 0), training window
x/D ≥ −2.5 and 0 ≤ z/D ≤ 4, random 80/20 split. GPU training is not
bit-reproducible; expect metrics close to, not identical with, those in
`data/05_neural_network`.

### ML-RANS solver

`code/03_solver/pimpleFoamReg_native` is the solver that produced all the
results in `data/06_solver_results`: the clustering and MLP inference run in
native C++ inside an OpenFOAM-v2212 PIMPLE solver with a modified k-omega SST
model (`kOmegaSSTML`). See its `README.md` for details. In short:

```bash
cd code/03_solver/pimpleFoamReg_native
python export_models.py      # models/ -> models/native_ml.dat (already provided)
wmake                        # with OpenFOAM v2212 sourced
```

- `models/` — K-means and MLP of `data/04_clustering` and
  `data/05_neural_network/ml_rans_clustering`, plus their export
  `native_ml.dat`; `models_noclustering/` — the no-clustering network.
- At every turbulence correction the k-omega SST equations are solved with the
  SST eddy viscosity, then ν<sub>t</sub> is replaced by the ML prediction;
  negative predictions are replaced by the molecular viscosity, 1.5e-5 m²/s.
  Each run covers 1.5 flow-through times of the domain (2520 m) with the
  K-means model and the MLP of `models/`, i.e. those obtained in training.
- For 8 and 15 m/s the inputs are rescaled to the training velocity
  (environment variable `NNCLUSTER_UINF_NORM`, factor r = 11.4/U<sub>∞</sub>).
- The cylindrical frame of the ML inputs is centred on the hub, as in
  training; `test/check_training_1T.py` checks that the solver frame
  reproduces the training clusters and scaler.
- `test/` contains the checks of the C++ inference against the
  Python/TensorFlow reference (agreement within 5e-6).
- `pimpleFoamReg_pt2` and `pimpleFoamReg_noclustering` are the earlier
  Python-embedded implementations, kept as the reference for those tests.

Some defaults in the solver and helper scripts point to the authors' cluster
(`/home/frank1/...`); override them with the environment variables listed in
the solver README (e.g. `PIMPLEFOAMREG_SOLVER_DIR`, `PIMPLEFOAMREG_MODEL_DIR`).

### LES post-processing

`code/01_les_postprocessing` documents how the training set was assembled:
`mergemesh.py` merges the two LES planes, `calcK.py` computes the resolved and
subgrid turbulent kinetic energy, and `interpolate_meshes.py` interpolates the
LES fields onto the RANS training planes. They are provided as a record of the
procedure; their file paths refer to the authors' working folders.

---

## Licence

- `code/03_solver/` (ML-RANS solver, derived from OpenFOAM): GNU General Public
  License v3.0 or later, see `code/03_solver/LICENSE`.
- All other files: BSD 3-Clause License, see `LICENSE`.

OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and
distributor of the OpenFOAM software via www.openfoam.com. This offering is not
approved or endorsed by OpenCFD Limited.

## Citation

If you use this code, please cite the article:

> A. Carloni, G. Delibra, F. De Girolamo, L. Tieghi, A. Corsini, V. F. Barnabei,
> *A Machine-Learning-Defined Eddy-Viscosity Closure for RANS Simulations of
> Horizontal-Axis Wind Turbine Wakes*, International Journal of Thermofluids
> (submitted).
