# === IMPORT & SETUP ===
import json
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pyvista as pv
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

start_time = time.time()

# === USER SWITCHES ===
PLOT = True

OPTUNA_TRIALS = int(os.environ.get("OPTUNA_TRIALS", "24"))
OPTUNA_EPOCHS = int(os.environ.get("OPTUNA_EPOCHS", "160"))
FINAL_EPOCHS = int(os.environ.get("FINAL_EPOCHS", "220"))
USE_OPTUNA = os.environ.get("USE_OPTUNA", "0").strip().lower() in {"1", "true", "yes", "y"}
MANUAL_PRESET = os.environ.get("MANUAL_PRESET", "wide_low_reg").strip()
LIVE_MONITOR = os.environ.get("LIVE_MONITOR", "1").strip().lower() in {"1", "true", "yes", "y"}
LIVE_METRICS_EVERY = int(os.environ.get("LIVE_METRICS_EVERY", "1"))
LIVE_PLOT_EVERY = int(os.environ.get("LIVE_PLOT_EVERY", "5"))

# Axial weighting:
# w_ax = 1 + slope * max(ax_d, 0)^power
# where ax_d = ax / ROTOR_DIAMETER_M.
ROTOR_DIAMETER_M = 126.0
TRAIN_MIN_AX_D = float(os.environ.get("TRAIN_MIN_AX_D", "-2.5"))
TRAIN_MIN_Z_D  = float(os.environ.get("TRAIN_MIN_Z_D", "0.0"))
TRAIN_MAX_Z_D  = float(os.environ.get("TRAIN_MAX_Z_D", "4.0"))
AXIAL_WEIGHT_SLOPE = float(os.environ.get("AXIAL_WEIGHT_SLOPE", "1.0"))
AXIAL_WEIGHT_POWER = float(os.environ.get("AXIAL_WEIGHT_POWER", "1.0"))
_ax_weight_max = os.environ.get("AXIAL_WEIGHT_MAX")
AXIAL_WEIGHT_MAX = float(_ax_weight_max) if _ax_weight_max else None

if TRAIN_MIN_Z_D >= TRAIN_MAX_Z_D:
    raise ValueError("TRAIN_MIN_Z_D must be < TRAIN_MAX_Z_D.")
if AXIAL_WEIGHT_SLOPE <= 0:
    raise ValueError("AXIAL_WEIGHT_SLOPE must be > 0.")
if AXIAL_WEIGHT_POWER <= 0:
    raise ValueError("AXIAL_WEIGHT_POWER must be > 0.")
if LIVE_METRICS_EVERY <= 0:
    raise ValueError("LIVE_METRICS_EVERY must be > 0.")
if LIVE_PLOT_EVERY <= 0:
    raise ValueError("LIVE_PLOT_EVERY must be > 0.")

# === PATHS ===
BASE_DIR = os.path.dirname(__file__)
PACKAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
BASE_DATA_PATH = os.path.join(PACKAGE_ROOT, "data")
# a fresh folder: if it already holds best_model_nut.keras, training is skipped
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", os.path.join(PACKAGE_ROOT, "retrained", "ml_rans_no_clustering"))
PREPROC_OUT = os.path.join(OUTPUT_ROOT, "preprocessing")
RESULTS_OUT = os.path.join(OUTPUT_ROOT, "results")
for folder in (OUTPUT_ROOT, PREPROC_OUT, RESULTS_OUT):
    os.makedirs(folder, exist_ok=True)

TRAIN_MESH_PATHS = [
    os.path.join(BASE_DATA_PATH, "03_training_data", "rans_mesh_with_les_data_.vtu")
]

VTK_OUTPUT = os.path.join(RESULTS_OUT, "rans_with_predictions.vtu")
MODEL_PATH = os.path.join(RESULTS_OUT, "best_model_nut.keras")
SKIP_TRAINING = os.path.exists(MODEL_PATH)
PREPROCESSOR_PATH = os.path.join(RESULTS_OUT, "preprocessor.pkl")
SCALER_PATH = os.path.join(RESULTS_OUT, "y_scaler_nut.pkl")
BEST_PARAMS_PATH = os.path.join(RESULTS_OUT, "best_params_nut.json")
HISTORY_PATH = os.path.join(RESULTS_OUT, "training_history_nut.csv")
HISTORY_PLOT_PATH = os.path.join(RESULTS_OUT, "training_validation_loss_nut.png")
METRICS_PATH = os.path.join(RESULTS_OUT, "metrics_nut.json")
LIVE_HISTORY_PATH = os.path.join(RESULTS_OUT, "training_live_metrics_nut.csv")
LIVE_PLOT_PATH = os.path.join(RESULTS_OUT, "training_live_metrics_nut.png")
TENSORBOARD_DIR = os.path.join(RESULTS_OUT, "tensorboard")
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

# === FEATURES ===
NUMERIC_FEATURES = ["Ux", "Ur", "Ut", "k", "nut", "z_d", "ax_d"]


# === MESH HELPERS ===

def read_mesh(path: str) -> pv.DataSet:
    """Read mesh, handling MultiBlock if needed."""
    mesh = pv.read(path)
    if isinstance(mesh, pv.MultiBlock):
        return mesh.combine()
    return mesh


def read_and_concat_meshes(paths: list[str]) -> pv.DataSet:
    """Read and merge multiple mesh files."""
    meshes = [read_mesh(p) for p in paths]
    merged = meshes[0]
    for mesh in meshes[1:]:
        merged = merged.merge(mesh, merge_points=True)
    return merged


def get_required_array(mesh: pv.DataSet, name: str, label: str | None = None) -> np.ndarray:
    """Return a required array from the mesh."""
    if name not in mesh.array_names:
        arr_label = label or name
        raise KeyError(f"Required array '{arr_label}' not found (expected '{name}').")
    return np.asarray(mesh[name])


def extract_velocity_components(mesh: pv.DataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract Ux, Ur, Ut from required scalar arrays."""
    Ux = get_required_array(mesh, "Uax", "Ux / Uax")
    Ur = get_required_array(mesh, "Ur", "Ur")
    Ut = get_required_array(mesh, "Ut", "Ut")
    return Ux, Ur, Ut


# === FEATURE BUILDING ===


def build_features(mesh: pv.DataSet):
    """
    Build feature DataFrame and extract target arrays.

    Returns:
        features_df: DataFrame with NUMERIC_FEATURES
        nut_rans: RANS viscosity (nut)
        nut_les: LES equivalent viscosity (nutEq)
    """
    Ux, Ur, Ut = extract_velocity_components(mesh)
    k_rans = get_required_array(mesh, "k", "k (RANS)")
    nut_rans = get_required_array(mesh, "nut", "nut (RANS)")
    nut_les = get_required_array(mesh, "nutEq", "nutEq (LES)")
    z_original = get_required_array(mesh, "z_original", "z_original")
    z_d = z_original / ROTOR_DIAMETER_M
    ax_m = get_required_array(mesh, "ax", "ax (axial distance in meters)")
    ax_d = ax_m / ROTOR_DIAMETER_M

    print(f"[INFO] z_d range: min={z_d.min():.4f}, max={z_d.max():.4f}")
    print(f"[INFO] ax_d range: min={ax_d.min():.4f}, max={ax_d.max():.4f}")

    features_df = pd.DataFrame({
        "Ux": Ux,
        "Ur": Ur,
        "Ut": Ut,
        "k": k_rans,
        "nut": nut_rans,
        "z_d": z_d,
        "ax_d": ax_d,
    })

    return features_df, nut_rans, nut_les


# === SAMPLE WEIGHTING ===


def compute_axial_weights(ax_d: np.ndarray) -> np.ndarray:
    """
    Axial weights in rotor diameters:
    - ax_d <= 0 -> weight 1
    - ax_d > 0  -> monotonic increasing weight
    """
    positive_ax = np.clip(np.asarray(ax_d, dtype=np.float64), 0.0, None)
    weights = 1.0 + AXIAL_WEIGHT_SLOPE * np.power(positive_ax, AXIAL_WEIGHT_POWER)
    if AXIAL_WEIGHT_MAX is not None:
        weights = np.minimum(weights, AXIAL_WEIGHT_MAX)
    return weights.astype(np.float32)


# === PLOTTING ===

def plot_distributions(nut_rans, nut_les, output_dir):
    """Plot initial nut distributions."""
    if not PLOT:
        return

    plt.figure(figsize=(8, 5))
    plt.hist(nut_rans, bins=100, density=True, histtype="step", color="steelblue", label=r"$\nu_t$ RANS")
    plt.hist(nut_les, bins=100, density=True, histtype="step", color="indianred", label=r"$\nu_{eq}$ LES")
    plt.xlabel(r"$\nu$")
    plt.ylabel("PDF")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nu_distributions.png"), dpi=300)
    plt.close()


def plot_training_history(history_csv, output_path):
    """Plot training history."""
    if not PLOT:
        return
    data = pd.read_csv(history_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(data["loss"], label="Train")
    axes[0].plot(data["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Loss")

    if "mae" in data.columns:
        axes[1].plot(data["mae"], label="Train MAE")
        axes[1].plot(data["val_mae"], label="Val MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE (normalized)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_title("MAE")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_predictions(y_true_scaled, y_pred_scaled, y_true_phys, y_pred_phys, output_dir):
    """Plot prediction diagnostics for nut."""
    if not PLOT:
        return

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_scaled, y_pred_scaled, s=1, alpha=0.4)
    lims = [min(y_true_scaled.min(), y_pred_scaled.min()), max(y_true_scaled.max(), y_pred_scaled.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel("Target (scaled)")
    plt.ylabel("Prediction (scaled)")
    plt.title("Normalized Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_true_scaled_nut.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_phys, y_pred_phys, s=1, alpha=0.4)
    lims = [min(y_true_phys.min(), y_pred_phys.min()), max(y_true_phys.max(), y_pred_phys.max())]
    plt.plot(lims, lims, "k--", lw=1)
    plt.xlabel(r"True $\nu_{eq}$")
    plt.ylabel(r"Predicted $\nu_{eq}$")
    plt.title("Physical Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pred_vs_true_physical_nut.png"), dpi=300)
    plt.close()


# === MODEL BUILDING ===

MANUAL_PARAM_PRESETS = {
    # Last Optuna result kept as a reproducible baseline.
    "optuna_previous": {
        "n_blocks": 6,
        "base_units": 144,
        "width_decay": 0.7247507621904381,
        "activation": "gelu",
        "use_layer_norm": True,
        "dropout": 0.0840262689341538,
        "l2_reg": 0.0005089523884209453,
        "learning_rate": 0.00010770196627287879,
        "huber_delta": 0.4102748821623355,
        "batch_size": 128,
        "residual": False,
    },
    # Default manual attempt: a little more capacity, lower regularization,
    # and a wider Huber quadratic region to penalize large misses more.
    "wide_low_reg": {
        "n_blocks": 7,
        "base_units": 192,
        "width_decay": 0.78,
        "activation": "gelu",
        "use_layer_norm": True,
        "dropout": 0.05,
        "l2_reg": 1e-4,
        "learning_rate": 1.5e-4,
        "huber_delta": 1.0,
        "batch_size": 128,
        "residual": True,
    },
    # Alternative if wide_low_reg starts to overfit.
    "deeper_regularized": {
        "n_blocks": 8,
        "base_units": 160,
        "width_decay": 0.82,
        "activation": "swish",
        "use_layer_norm": True,
        "dropout": 0.08,
        "l2_reg": 2e-4,
        "learning_rate": 1.0e-4,
        "huber_delta": 0.8,
        "batch_size": 128,
        "residual": True,
    },
    # Alternative if training is slow or noisy.
    "compact_fast": {
        "n_blocks": 5,
        "base_units": 160,
        "width_decay": 0.75,
        "activation": "gelu",
        "use_layer_norm": True,
        "dropout": 0.04,
        "l2_reg": 1e-4,
        "learning_rate": 2.0e-4,
        "huber_delta": 0.8,
        "batch_size": 256,
        "residual": False,
    },
}


def get_manual_params(preset_name: str) -> dict:
    """Return fixed hyperparameters for a manual training run."""
    if preset_name not in MANUAL_PARAM_PRESETS:
        available = ", ".join(sorted(MANUAL_PARAM_PRESETS))
        raise ValueError(f"Unknown MANUAL_PRESET='{preset_name}'. Available presets: {available}")

    params = dict(MANUAL_PARAM_PRESETS[preset_name])

    params_json = os.environ.get("MANUAL_PARAMS_JSON")
    if params_json:
        overrides = json.loads(params_json)
        unknown = sorted(set(overrides) - set(params))
        if unknown:
            raise ValueError(f"Unknown manual parameter override(s): {unknown}")
        params.update(overrides)

    return params


def suggest_params(trial) -> dict:
    """Suggest hyperparameters for Optuna trial."""
    return {
        # Deeper-but-narrower prior to reduce overfitting.
        "n_blocks": trial.suggest_int("n_blocks", 5, 8),
        "base_units": trial.suggest_int("base_units", 128, 224, step=16),
        "width_decay": trial.suggest_float("width_decay", 0.72, 0.86),
        "activation": trial.suggest_categorical("activation", ["relu", "elu", "swish", "gelu"]),
        "use_layer_norm": trial.suggest_categorical("use_layer_norm", [True, False]),
        "dropout": trial.suggest_float("dropout", 0.02, 0.18),
        "l2_reg": trial.suggest_float("l2_reg", 1e-5, 1e-3, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 4e-4, log=True),
        "huber_delta": trial.suggest_float("huber_delta", 0.4, 1.5),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "residual": trial.suggest_categorical("residual", [True, False]),
    }


def build_model(params: dict, input_dim: int) -> keras.Model:
    """Build and compile single-output model for delta_nut."""
    reg = keras.regularizers.l2(params["l2_reg"])

    inputs = keras.Input(shape=(input_dim,), name="features")
    x = inputs

    widths = []
    current = params["base_units"]
    for _ in range(params["n_blocks"]):
        widths.append(max(int(round(current)), 32))
        current *= params["width_decay"]

    for width in widths:
        shortcut = x
        x = keras.layers.Dense(width, kernel_regularizer=reg, kernel_initializer="he_normal")(x)
        if params["use_layer_norm"]:
            x = keras.layers.LayerNormalization()(x)
        x = keras.layers.Activation(params["activation"])(x)
        if params["dropout"] > 0:
            x = keras.layers.Dropout(params["dropout"])(x)
        if params.get("residual", False):
            if shortcut.shape[-1] != x.shape[-1]:
                shortcut = keras.layers.Dense(
                    width,
                    kernel_regularizer=reg,
                    kernel_initializer="glorot_normal",
                )(shortcut)
            x = keras.layers.Add()([x, shortcut])

    outputs = keras.layers.Dense(1, kernel_initializer="glorot_normal")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss=keras.losses.Huber(delta=params["huber_delta"]),
        weighted_metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


# === LIVE MONITORING ===

def _nan_if_empty(values, reducer):
    """Apply a reducer or return NaN for empty arrays."""
    values = np.asarray(values)
    if values.size == 0:
        return np.nan
    return float(reducer(values))


def _r2_or_nan(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 that returns NaN when the subset has no variance."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2:
        return np.nan
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return np.nan
    return float(1.0 - np.sum((y_pred - y_true) ** 2) / ss_tot)


def plot_live_training_metrics(history_rows: list[dict], output_path: str):
    """Write an updating PNG with the most useful training diagnostics."""
    if not PLOT or not history_rows:
        return

    data = pd.DataFrame(history_rows)
    epochs = data["epoch"].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    def plot_columns(ax, columns, title, y_label, yscale=None):
        plotted = False
        for column, label in columns:
            if column not in data:
                continue
            values = data[column].to_numpy(dtype=np.float64)
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            ax.plot(epochs[mask], values[mask], label=label)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(y_label)
        if yscale:
            ax.set_yscale(yscale)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()

    plot_columns(
        axes[0],
        [("loss", "train"), ("val_loss", "validation")],
        "Loss",
        "Huber loss",
        yscale="log",
    )
    plot_columns(
        axes[1],
        [("mae", "train"), ("val_mae", "validation")],
        "Scaled MAE",
        "MAE",
    )
    plot_columns(
        axes[2],
        [("rmse", "train"), ("val_rmse", "validation")],
        "Scaled RMSE",
        "RMSE",
    )
    plot_columns(
        axes[3],
        [("val_phys_mae", "phys MAE"), ("val_phys_rmse", "phys RMSE"), ("val_tail_ge_10_mae", "tail >=10 MAE")],
        "Validation Physical Errors",
        r"$\nu$ error",
    )
    plot_columns(
        axes[4],
        [("val_phys_r2", "R2"), ("val_phys_bias", "bias")],
        "Validation Physical R2 / Bias",
        "value",
    )
    plot_columns(
        axes[5],
        [("val_negative_pred_pct", "negative pred %"), ("val_phys_p95_ae", "P95 AE"), ("val_phys_p99_ae", "P99 AE")],
        "Validation Risk Indicators",
        "value",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


class LiveValidationMetricsCallback(keras.callbacks.Callback):
    """Compute physical validation diagnostics during training."""

    def __init__(
        self,
        X_val: np.ndarray,
        y_scaler: StandardScaler,
        nut_rans_val: np.ndarray,
        nut_les_val: np.ndarray,
        every: int,
        plot_every: int,
        live_plot_path: str,
    ):
        super().__init__()
        self.X_val = X_val
        self.y_scaler = y_scaler
        self.nut_rans_val = np.asarray(nut_rans_val, dtype=np.float64)
        self.nut_les_val = np.asarray(nut_les_val, dtype=np.float64)
        self.every = every
        self.plot_every = plot_every
        self.live_plot_path = live_plot_path
        self.rows = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        epoch_number = epoch + 1
        row = {"epoch": epoch_number}
        row.update({key: float(value) for key, value in logs.items() if np.isscalar(value)})

        if epoch_number == 1 or epoch_number % self.every == 0:
            y_pred_scaled = self.model.predict(self.X_val, verbose=0).reshape(-1)
            delta_pred = self.y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)
            nut_pred = self.nut_rans_val + delta_pred
            err = nut_pred - self.nut_les_val
            abs_err = np.abs(err)

            phys_metrics = {
                "val_phys_r2": _r2_or_nan(self.nut_les_val, nut_pred),
                "val_phys_mae": float(np.mean(abs_err)),
                "val_phys_rmse": float(np.sqrt(np.mean(err**2))),
                "val_phys_bias": float(np.mean(err)),
                "val_phys_p95_ae": float(np.quantile(abs_err, 0.95)),
                "val_phys_p99_ae": float(np.quantile(abs_err, 0.99)),
                "val_phys_max_ae": float(np.max(abs_err)),
                "val_negative_pred_pct": float(100.0 * np.mean(nut_pred < 0.0)),
                "val_pred_min": float(np.min(nut_pred)),
                "val_pred_max": float(np.max(nut_pred)),
            }

            for threshold in (5.0, 10.0, 15.0):
                mask = self.nut_les_val >= threshold
                label = f"val_tail_ge_{threshold:g}"
                phys_metrics[f"{label}_count"] = int(np.sum(mask))
                phys_metrics[f"{label}_mae"] = _nan_if_empty(abs_err[mask], np.mean)
                phys_metrics[f"{label}_rmse"] = _nan_if_empty(err[mask], lambda x: np.sqrt(np.mean(x**2)))
                phys_metrics[f"{label}_r2"] = _r2_or_nan(self.nut_les_val[mask], nut_pred[mask])

            logs.update(phys_metrics)
            row.update(phys_metrics)

            print(
                "\n"
                f"[MONITOR] epoch={epoch_number} "
                f"val_phys_R2={phys_metrics['val_phys_r2']:.4f} "
                f"val_phys_MAE={phys_metrics['val_phys_mae']:.4f} "
                f"val_phys_RMSE={phys_metrics['val_phys_rmse']:.4f} "
                f"tail>=10_MAE={phys_metrics['val_tail_ge_10_mae']:.4f} "
                f"neg%={phys_metrics['val_negative_pred_pct']:.2f}"
                "\n",
                flush=True,
            )

        self.rows.append(row)
        if epoch_number % self.plot_every == 0 or epoch_number == 1:
            plot_live_training_metrics(self.rows, self.live_plot_path)


# === TRAINING ===

def run_optuna(X_train, y_train, X_val, y_val, sw_train, sw_val, input_dim) -> optuna.Study:
    """Run Optuna hyperparameter optimization."""

    def objective(trial):
        params = suggest_params(trial)
        model = build_model(params, input_dim)
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=10,
                monitor="val_loss",
                restore_best_weights=True,
                min_delta=1e-4,
            ),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val, sw_val),
            epochs=OPTUNA_EPOCHS,
            batch_size=params["batch_size"],
            sample_weight=sw_train,
            callbacks=callbacks,
            verbose=0,
        )
        return float(np.min(history.history["val_loss"]))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, n_jobs=1)
    return study


def train_final_model(params, X_train, y_train, X_val, y_val, sw_train, sw_val, input_dim, live_data=None):
    """Train final model with best parameters."""
    model = build_model(params, input_dim)
    print(f"  Model parameters: {model.count_params():,}")
    callbacks = [
        keras.callbacks.TerminateOnNaN(),
    ]
    if LIVE_MONITOR and live_data is not None:
        callbacks.extend(
            [
                LiveValidationMetricsCallback(
                    X_val=X_val,
                    y_scaler=live_data["y_scaler"],
                    nut_rans_val=live_data["nut_rans_val"],
                    nut_les_val=live_data["nut_les_val"],
                    every=LIVE_METRICS_EVERY,
                    plot_every=LIVE_PLOT_EVERY,
                    live_plot_path=LIVE_PLOT_PATH,
                ),
                keras.callbacks.CSVLogger(LIVE_HISTORY_PATH, append=False),
                keras.callbacks.TensorBoard(
                    log_dir=TENSORBOARD_DIR,
                    histogram_freq=0,
                    write_graph=True,
                    update_freq="epoch",
                ),
            ]
        )
        print(f"  Live CSV metrics: {LIVE_HISTORY_PATH}")
        print(f"  Live plot: {LIVE_PLOT_PATH}")
        print(f"  TensorBoard logs: {TENSORBOARD_DIR}")
    callbacks.extend([
        keras.callbacks.EarlyStopping(
            patience=20,
            monitor="val_loss",
            restore_best_weights=True,
            min_delta=1e-5,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1,
        ),
    ])
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val, sw_val),
        epochs=FINAL_EPOCHS,
        batch_size=params["batch_size"],
        sample_weight=sw_train,
        callbacks=callbacks,
        verbose=1,
    )
    return model, history


# === EVALUATION METRICS ===

def _finite_float(value) -> float | None:
    """Return a JSON-safe float, using None for non-finite values."""
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def regression_metrics(y_true, y_pred) -> dict:
    """Compute robust regression diagnostics without sample weighting."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    n = int(y_true.size)
    if n == 0:
        return {
            "n": 0,
            "r2": None,
            "mae": None,
            "rmse": None,
            "bias": None,
            "median_ae": None,
            "p90_ae": None,
            "p95_ae": None,
            "p99_ae": None,
            "max_ae": None,
            "pred_min": None,
            "pred_max": None,
            "true_min": None,
            "true_max": None,
            "negative_pred_pct": None,
        }

    err = y_pred - y_true
    abs_err = np.abs(err)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = None if ss_tot <= 0 else 1.0 - np.sum(err**2) / ss_tot

    return {
        "n": n,
        "r2": _finite_float(r2) if r2 is not None else None,
        "mae": _finite_float(np.mean(abs_err)),
        "rmse": _finite_float(np.sqrt(np.mean(err**2))),
        "bias": _finite_float(np.mean(err)),
        "median_ae": _finite_float(np.median(abs_err)),
        "p90_ae": _finite_float(np.quantile(abs_err, 0.90)),
        "p95_ae": _finite_float(np.quantile(abs_err, 0.95)),
        "p99_ae": _finite_float(np.quantile(abs_err, 0.99)),
        "max_ae": _finite_float(np.max(abs_err)),
        "pred_min": _finite_float(np.min(y_pred)),
        "pred_max": _finite_float(np.max(y_pred)),
        "true_min": _finite_float(np.min(y_true)),
        "true_max": _finite_float(np.max(y_true)),
        "negative_pred_pct": _finite_float(100.0 * np.mean(y_pred < 0.0)),
    }


def _fmt_metric(value, digits=4) -> str:
    """Format a metric that may be missing."""
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def print_metric_line(name: str, metrics: dict):
    """Print a compact metric row."""
    print(
        f"  {name:<12} n={metrics['n']:>6d} "
        f"R2={_fmt_metric(metrics['r2'])} "
        f"MAE={_fmt_metric(metrics['mae'])} "
        f"RMSE={_fmt_metric(metrics['rmse'])} "
        f"bias={_fmt_metric(metrics['bias'])} "
        f"P95AE={_fmt_metric(metrics['p95_ae'])} "
        f"P99AE={_fmt_metric(metrics['p99_ae'])} "
        f"maxAE={_fmt_metric(metrics['max_ae'])} "
        f"neg%={_fmt_metric(metrics['negative_pred_pct'], 2)}"
    )


def mask_to_indices(mask: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to integer indices."""
    return np.flatnonzero(np.asarray(mask, dtype=bool))


def evaluate_model_predictions(
    y_true_scaled: np.ndarray,
    y_pred_scaled: np.ndarray,
    nut_les: np.ndarray,
    nut_pred: np.ndarray,
    ax_d: np.ndarray,
    z_d: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
) -> dict:
    """Print and return detailed model diagnostics."""
    n_all = len(nut_les)
    all_indices = np.arange(n_all)

    subsets = {
        "train": np.asarray(train_indices, dtype=np.int64),
        "validation": np.asarray(val_indices, dtype=np.int64),
        "all": all_indices,
    }

    report = {
        "normalized_delta": {},
        "physical_nut": {},
        "validation_by_true_threshold": {},
        "validation_by_ax_d": {},
        "validation_by_z_d": {},
        "worst_validation_errors": [],
    }

    print("\n" + "=" * 72)
    print("DETAILED METRICS")
    print("=" * 72)

    print("\nNormalized delta metrics (network target):")
    for name, idx in subsets.items():
        metrics = regression_metrics(y_true_scaled[idx], y_pred_scaled[idx])
        report["normalized_delta"][name] = metrics
        print_metric_line(name, metrics)

    print("\nPhysical metrics: nut_pred vs nutEq LES")
    for name, idx in subsets.items():
        metrics = regression_metrics(nut_les[idx], nut_pred[idx])
        report["physical_nut"][name] = metrics
        print_metric_line(name, metrics)

    val_mask = np.zeros(n_all, dtype=bool)
    val_mask[val_indices] = True

    print("\nValidation tail metrics by true nutEq threshold:")
    for threshold in (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0):
        mask = val_mask & (nut_les >= threshold)
        idx = mask_to_indices(mask)
        metrics = regression_metrics(nut_les[idx], nut_pred[idx])
        report["validation_by_true_threshold"][f"true_ge_{threshold:g}"] = metrics
        print_metric_line(f">= {threshold:g}", metrics)

    print("\nValidation metrics by ax_d range:")
    ax_bins = [-np.inf, 0.0, 1.0, 2.0, 4.0, 8.0, 12.0, np.inf]
    for lo, hi in zip(ax_bins[:-1], ax_bins[1:]):
        mask = val_mask & (ax_d >= lo) & (ax_d < hi)
        idx = mask_to_indices(mask)
        label = f"[{lo:g},{hi:g})"
        metrics = regression_metrics(nut_les[idx], nut_pred[idx])
        report["validation_by_ax_d"][label] = metrics
        print_metric_line(label, metrics)

    print("\nValidation metrics by z_d range:")
    z_bins = [-np.inf, 0.5, 1.0, 2.0, 4.0, 6.0, np.inf]
    for lo, hi in zip(z_bins[:-1], z_bins[1:]):
        mask = val_mask & (z_d >= lo) & (z_d < hi)
        idx = mask_to_indices(mask)
        label = f"[{lo:g},{hi:g})"
        metrics = regression_metrics(nut_les[idx], nut_pred[idx])
        report["validation_by_z_d"][label] = metrics
        print_metric_line(label, metrics)

    val_err = nut_pred[val_indices] - nut_les[val_indices]
    order = np.argsort(np.abs(val_err))[-10:][::-1]
    print("\nWorst validation absolute errors:")
    print("  rank global_idx ax_d z_d true pred error")
    for rank, local_pos in enumerate(order, start=1):
        global_idx = int(val_indices[local_pos])
        row = {
            "rank": rank,
            "global_idx": global_idx,
            "ax_d": _finite_float(ax_d[global_idx]),
            "z_d": _finite_float(z_d[global_idx]),
            "true_nutEq": _finite_float(nut_les[global_idx]),
            "pred_nut": _finite_float(nut_pred[global_idx]),
            "error": _finite_float(nut_pred[global_idx] - nut_les[global_idx]),
        }
        report["worst_validation_errors"].append(row)
        print(
            f"  {rank:>4d} {global_idx:>10d} "
            f"{row['ax_d']:>6.3f} {row['z_d']:>6.3f} "
            f"{row['true_nutEq']:>8.3f} {row['pred_nut']:>8.3f} "
            f"{row['error']:>8.3f}"
        )

    return report


# === MAIN ===

def main():
    print("=" * 72)
    print("Neural Network Training (single output: delta_nut, no clustering inputs)")
    print("=" * 72)
    print(f"[INFO] USE_OPTUNA = {USE_OPTUNA}")
    if not USE_OPTUNA:
        print(f"[INFO] Manual hyperparameter preset = {MANUAL_PRESET}")
    print(
        f"[INFO] LIVE_MONITOR = {LIVE_MONITOR} "
        f"(metrics every {LIVE_METRICS_EVERY} epoch(s), plot every {LIVE_PLOT_EVERY} epoch(s))"
    )
    print(f"[INFO] Rotor diameter D = {ROTOR_DIAMETER_M} m")
    print("[INFO] Coordinate features normalized in diameters: z_d = z_original / D, ax_d = ax / D")
    print(f"[INFO] Training/validation axial filter: ax_d >= {TRAIN_MIN_AX_D:g}")
    print(f"[INFO] Training/validation radial filter: {TRAIN_MIN_Z_D:g} <= z_d <= {TRAIN_MAX_Z_D:g}")
    print(
        "[INFO] Axial weight formula: "
        f"w_ax = 1 + {AXIAL_WEIGHT_SLOPE} * max(ax_d, 0)^{AXIAL_WEIGHT_POWER}"
    )
    if AXIAL_WEIGHT_MAX is not None:
        print(f"[INFO] Axial weight cap: {AXIAL_WEIGHT_MAX}")

    # --- 1. Load mesh ---
    print("\n[1/7] Loading mesh...")
    mesh_rans = read_and_concat_meshes(TRAIN_MESH_PATHS).cell_data_to_point_data()
    print(f"  Mesh points: {mesh_rans.n_points}")

    # --- 2. Build features ---
    print("\n[2/7] Building features...")
    features_df, nut_rans, nut_les = build_features(mesh_rans)
    delta = nut_les - nut_rans
    print(f"  Delta nut range: [{delta.min():.4f}, {delta.max():.4f}]")
    plot_distributions(nut_rans, nut_les, PREPROC_OUT)

    # --- 3. Train/Val split ---
    print("\n[3/7] Splitting data...")
    ax_d_all = features_df["ax_d"].to_numpy()
    z_d_all  = features_df["z_d"].to_numpy()
    trainval_mask = (
        (ax_d_all >= TRAIN_MIN_AX_D) &
        (z_d_all  >= TRAIN_MIN_Z_D)  &
        (z_d_all  <= TRAIN_MAX_Z_D)
    )
    excluded_count = int(np.sum(~trainval_mask))
    if not np.any(trainval_mask):
        raise ValueError(
            f"Training filter ax_d >= {TRAIN_MIN_AX_D:g}, "
            f"{TRAIN_MIN_Z_D:g} <= z_d <= {TRAIN_MAX_Z_D:g} removed all points."
        )
    print(
        f"  Training/validation points kept: {int(np.sum(trainval_mask))} / {len(trainval_mask)} "
        f"(excluded: {excluded_count})"
    )

    features_trainval = features_df.loc[trainval_mask]
    delta_trainval = delta[trainval_mask]
    ax_d_trainval = ax_d_all[trainval_mask]
    split = train_test_split(
        features_trainval,
        delta_trainval,
        ax_d_trainval,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_val, y_train_raw, y_val_raw, ax_d_train, ax_d_val = split
    train_indices = X_train.index.to_numpy(dtype=np.int64)
    val_indices = X_val.index.to_numpy(dtype=np.int64)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")

    # --- 4. Preprocessing ---
    print("\n[4/7] Preprocessing features and targets...")
    if SKIP_TRAINING:
        print("  Loading saved preprocessor and scaler (SKIP_TRAINING=True)...")
        with open(PREPROCESSOR_PATH, "rb") as f:
            preprocessor = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            y_scaler = pickle.load(f)
        X_train_prep = preprocessor.transform(X_train).astype(np.float32)
        X_val_prep   = preprocessor.transform(X_val).astype(np.float32)
        X_all_prep   = preprocessor.transform(features_df).astype(np.float32)
        input_dim = X_train_prep.shape[1]
        y_train_scaled = y_scaler.transform(np.asarray(y_train_raw).reshape(-1, 1)).flatten().astype(np.float32)
        y_val_scaled   = y_scaler.transform(np.asarray(y_val_raw).reshape(-1, 1)).flatten().astype(np.float32)
        y_all_scaled   = y_scaler.transform(delta.reshape(-1, 1)).flatten().astype(np.float32)
        print(f"  Input dimension: {input_dim}")
        print(f"  Target scale: {y_scaler.scale_[0]:.4f}, mean: {y_scaler.mean_[0]:.4f}")
        with open(BEST_PARAMS_PATH) as f:
            best_params = json.load(f)
        print(f"  Loaded best_params: {best_params}")
    else:
        preprocessor = ColumnTransformer(
            [
                ("num", StandardScaler(), NUMERIC_FEATURES),
            ],
            remainder="drop",
        )
        preprocessor.fit(X_train)
        X_train_prep = preprocessor.transform(X_train).astype(np.float32)
        X_val_prep = preprocessor.transform(X_val).astype(np.float32)
        X_all_prep = preprocessor.transform(features_df).astype(np.float32)
        input_dim = X_train_prep.shape[1]
        print(f"  Input dimension: {input_dim}")

        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(np.asarray(y_train_raw).reshape(-1, 1)).flatten().astype(np.float32)
        y_val_scaled = y_scaler.transform(np.asarray(y_val_raw).reshape(-1, 1)).flatten().astype(np.float32)
        y_all_scaled = y_scaler.transform(delta.reshape(-1, 1)).flatten().astype(np.float32)
        print(f"  Target scale: {y_scaler.scale_[0]:.4f}, mean: {y_scaler.mean_[0]:.4f}")

        with open(PREPROCESSOR_PATH, "wb") as f:
            pickle.dump(preprocessor, f)

        sw_ax_train = compute_axial_weights(ax_d_train)
        sw_ax_val = compute_axial_weights(ax_d_val)
        sw_train = sw_ax_train
        sw_val = sw_ax_val

        print(
            f"  Sample weight stats, axial only (train): min={sw_train.min():.4f}, "
            f"mean={sw_train.mean():.4f}, max={sw_train.max():.4f}"
        )
        print(
            f"  Axial factor stats (train): min={sw_ax_train.min():.4f}, "
            f"mean={sw_ax_train.mean():.4f}, max={sw_ax_train.max():.4f}"
        )

        # --- 5. Hyperparameters ---
        print("\n[5/7] Selecting hyperparameters...")
        if USE_OPTUNA:
            print("  Running Optuna optimization...")
            study = run_optuna(X_train_prep, y_train_scaled, X_val_prep, y_val_scaled, sw_train, sw_val, input_dim)
            best_params = dict(study.best_trial.params)
            print(f"  Best trial: {study.best_trial.number}")
            print(f"  Best val_loss: {study.best_value:.6f}")
        else:
            print("  Skipping Optuna.")
            best_params = get_manual_params(MANUAL_PRESET)
        print(f"  Selected params: {best_params}")
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)

        # --- 6. Final training ---
        print("\n[6/7] Training final model...")
        live_data = {
            "y_scaler": y_scaler,
            "nut_rans_val": nut_rans[val_indices],
            "nut_les_val": nut_les[val_indices],
        }
        model, history = train_final_model(
            best_params,
            X_train_prep,
            y_train_scaled,
            X_val_prep,
            y_val_scaled,
            sw_train,
            sw_val,
            input_dim,
            live_data=live_data,
        )
        model.save(MODEL_PATH)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(y_scaler, f)
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(HISTORY_PATH, index=False)
        plot_training_history(HISTORY_PATH, HISTORY_PLOT_PATH)
        best_epoch = int(history_df["val_loss"].idxmin()) + 1
        print(
            f"  Best epoch: {best_epoch} | "
            f"val_loss={history_df['val_loss'].min():.6f} | "
            f"val_mae={history_df['val_mae'].min():.6f}"
        )
        if "val_rmse" in history_df.columns:
            print(f"  Best val_rmse: {history_df['val_rmse'].min():.6f}")

    if SKIP_TRAINING:
        print("\n[5-6/7] Skipped (model loaded from disk).")
        model = keras.models.load_model(MODEL_PATH)
        print(f"  Model loaded from: {MODEL_PATH}")

    # --- 7. Evaluation and output ---
    print("\n[7/7] Evaluating model and writing VTK...")
    y_pred_scaled = model.predict(X_all_prep, verbose=0).flatten()
    delta_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    nut_pred = nut_rans + delta_pred

    r2_scaled = r2_score(y_all_scaled, y_pred_scaled)
    mae_scaled = np.mean(np.abs(y_all_scaled - y_pred_scaled))
    r2_phys = r2_score(nut_les, nut_pred)
    mse_phys = mean_squared_error(nut_les, nut_pred)
    mae_phys = np.mean(np.abs(nut_les - nut_pred))

    print("\n" + "=" * 40)
    print("RESULTS (nut)")
    print("=" * 40)
    print("Normalized space:")
    print(f"  R² = {r2_scaled:.4f}")
    print(f"  MAE = {mae_scaled:.4f}")
    print("Physical space:")
    print(f"  R² = {r2_phys:.4f}")
    print(f"  MSE = {mse_phys:.4e}")
    print(f"  MAE = {mae_phys:.4f}")

    metrics_report = evaluate_model_predictions(
        y_true_scaled=y_all_scaled,
        y_pred_scaled=y_pred_scaled,
        nut_les=nut_les,
        nut_pred=nut_pred,
        ax_d=features_df["ax_d"].to_numpy(),
        z_d=features_df["z_d"].to_numpy(),
        train_indices=train_indices,
        val_indices=val_indices,
    )
    metrics_report["run_config"] = {
        "use_optuna": USE_OPTUNA,
        "manual_preset": None if USE_OPTUNA else MANUAL_PRESET,
        "params": best_params,
        "final_epochs": FINAL_EPOCHS,
        "optuna_trials": OPTUNA_TRIALS if USE_OPTUNA else 0,
        "optuna_epochs": OPTUNA_EPOCHS if USE_OPTUNA else 0,
        "train_min_ax_d": TRAIN_MIN_AX_D,
        "train_min_z_d": TRAIN_MIN_Z_D,
        "train_max_z_d": TRAIN_MAX_Z_D,
        "trainval_points": int(np.sum(trainval_mask)),
        "excluded_inflow_points": excluded_count,
        "axial_weight_slope": AXIAL_WEIGHT_SLOPE,
        "axial_weight_power": AXIAL_WEIGHT_POWER,
        "axial_weight_max": AXIAL_WEIGHT_MAX,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_report, f, indent=2)
    print(f"\nDetailed metrics saved to: {METRICS_PATH}")

    plot_predictions(y_all_scaled, y_pred_scaled, nut_les, nut_pred, RESULTS_OUT)

    mesh_rans["nut_pred"] = nut_pred
    mesh_rans["delta_pred"] = delta_pred
    mesh_rans["delta_nut_pred"] = delta_pred
    mesh_rans["z_d"] = features_df["z_d"].to_numpy()
    mesh_rans["ax_d"] = features_df["ax_d"].to_numpy()
    mesh_rans.save(VTK_OUTPUT)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
