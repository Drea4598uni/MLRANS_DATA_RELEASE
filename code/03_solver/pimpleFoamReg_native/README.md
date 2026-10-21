# pimpleFoamReg_native

ML-RANS solver of the paper: an OpenFOAM-v2212 PIMPLE solver
(`pimpleFoamRegNative`) with the turbulence model `kOmegaSSTML`, in which the
eddy viscosity entering the momentum equations is given by the K-means
clustering and the MLP trained on the LES. Clustering and MLP inference run in
native C++: no Python or TensorFlow is needed at run time.

## How it works

At every turbulence correction:

1. the SST eddy viscosity of the previous step is restored and the k and ω
   equations of the k-omega SST model are solved;
2. for every cell the inputs are computed in the cylindrical frame of the
   rotor, centred on the hub as in training (geometry read from the case's
   `fvOptions` and `Trim` dictionaries);
3. the cell is assigned to the nearest K-means centroid in the min-max scaled
   space (U<sub>x</sub>, U<sub>θ</sub>, U<sub>r</sub>), with the cluster labels
   ordered as in training;
4. the MLP predicts the correction Δν<sub>t</sub>, and
   ν<sub>t,ML</sub> = ν<sub>t</sub> + Δν<sub>t</sub> replaces the eddy
   viscosity in the cells; negative values are replaced by the molecular
   viscosity (1.5e-5 m²/s); at the ground the wall-face values are given by the
   wall function of the case.

The solver writes, besides the usual fields, the cluster label `clusterID`,
the rotor-frame coordinates `ax`, `rho`, `theta` and the SST eddy viscosity
before the correction `nut_original`.

## Files

| File | Content |
|---|---|
| `pimpleFoamReg.C`, `createFields.H`, `UEqn.H`, `pEqn.H`, ... | PIMPLE solver with the call to the ML correction |
| `kOmegaSSTML/` | k-omega SST model used by the solver |
| `nativeML.H` | frame transformation, scalers, K-means assignment and MLP forward pass (single precision, AVX-512 kernels with scalar fallback, OpenMP) |
| `export_models.py` | converts the trained models into `native_ml.dat` |
| `models/` | K-means model, scalers and MLP of the paper (`km`, `km-scaler`, `preprocessor.pkl`, `y_scaler_nut.pkl`, `best_model_nut.keras`) and their export `native_ml.dat` |
| `models_noclustering/` | export of the network trained without clustering |
| `test/` | checks of the native implementation (see below) |

## Use

1. Export of the model parameters (already provided in `models*/native_ml.dat`;
   needed only for retrained models), with the environment of
   `requirements-training.txt`:

   ```bash
   python export_models.py      # options: --model-dir, --model-file, --scaler-file, --output
   ```

   The network is exported as a generic graph of layers (Dense, normalisation
   layers, element-wise merges, the common activation functions); scalers are
   exported as affine maps and checked numerically. An unsupported layer,
   activation or scaler stops the export with an explicit message.

2. Compilation, with OpenFOAM v2212 sourced:

   ```bash
   wmake                        # -> $FOAM_USER_APPBIN/pimpleFoamRegNative
   ```

   `Make/options` compiles with `-march=native -fopenmp`.

3. Run: set `application pimpleFoamRegNative;` in the case's `controlDict`
   and run it as any OpenFOAM solver (serial or MPI).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PIMPLEFOAMREG_SOLVER_DIR` | built-in path | folder of the solver |
| `PIMPLEFOAMREG_MODEL_DIR` | `<solver dir>/models` | folder of the models |
| `PIMPLEFOAMREG_NATIVE_PARAMS` | `<model dir>/native_ml.dat` | exported parameter file |
| `NNTURBINE` | first turbine in `fvOptions` | turbine that defines the frame |
| `NNCORR_ROTOR_DIAMETER` | 126.0 | diameter used for z/D and x/D |
| `NNCORR_TARGET_MODE` | from the file name (`delta`) | `delta` (correction) or `absolute` target |
| `NNCORR_TARGET_TRANSFORM` | from the file name (`none`) | `none` or `log` |
| `NNCLUSTER_REORDER_VAR`, `NNCLUSTER_REORDER_ASCENDING` | `Ux`, false | ordering of the cluster labels |
| `NNCLUSTER_UINF_NORM`, `NNCLUSTER_UINF_TRAIN`, `NNCLUSTER_UINF_INFER` | off, –, `fvOptions` value | rescaling of the inputs to the training velocity, r = U<sub>train</sub>/U<sub>∞</sub> (inputs U·r, k·r², ν<sub>t</sub>·r, output ÷ r); used with 11.4 and 8 or 15 for the runs at 8 and 15 m/s |
| `NATIVE_ML_THREADS` | 1 | OpenMP threads of the inference per MPI rank |

With MPI runs keep `NATIVE_ML_THREADS=1`, the parallelism being given by the
domain decomposition.

## Tests (`test/`)

- `gen_reference.py`, `gen_reference_noclustering.py`, `testNative.C`,
  `compare.py`: the complete native pipeline against the Python/TensorFlow
  implementation of `../pimpleFoamReg_pt2` on 20 000 synthetic inputs; the
  frame and the clustering agree to machine precision (identical cluster
  labels) and the corrected ν<sub>t</sub> within 5e-6 (single-precision
  round-off).
- `testMLP.C`, `test_robustness.py`: export and native forward pass of eight
  different network architectures against Keras (agreement within ~1.5e-6).
- `geomCheck.C`, `geom_compare.py`: geometry parsing on real cases.
- `check_training_1T.py`: the solver frame reproduces the cluster labels and
  the scaler of the training data.
- `benchPipeline.C`: timing of the stages of the pipeline.

Test programs: `g++ -std=c++17 -O3 -march=native -fopenmp -o <name> <name>.C`.
Some test scripts refer to the authors' folders for the case data; pass the
paths of your own copies.
