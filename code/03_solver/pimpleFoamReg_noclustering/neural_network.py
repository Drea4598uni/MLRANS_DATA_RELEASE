import numpy as np
import os
import warnings
import json
import re
import tempfile
import zipfile
from typing import Optional, Tuple, List, Dict, Any

# Configurazione MPI e TensorFlow PRIMA di importare TensorFlow
def get_mpi_rank():
    """Ottieni il rank MPI corrente. Ritorna 0 se non in ambiente MPI."""
    try:
        # Prova diversi metodi per ottenere il rank MPI
        if 'PMI_RANK' in os.environ:
            return int(os.environ['PMI_RANK'])
        elif 'OMPI_COMM_WORLD_RANK' in os.environ:
            return int(os.environ['OMPI_COMM_WORLD_RANK'])
        elif 'MV2_COMM_WORLD_RANK' in os.environ:
            return int(os.environ['MV2_COMM_WORLD_RANK'])
        else:
            return 0
    except:
        return 0

def is_master_process():
    """Controlla se questo è il processo master (rank 0)"""
    return get_mpi_rank() == 0

# MEMORY OPTIMIZATION: Setup TensorFlow logging PRIMA dell'import
if not is_master_process():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Solo FATAL
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disabilita messaggi oneDNN
    os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
    os.environ['TF_SILENCE_CUDA_DEVICE_PLACEMENT_WARNINGS'] = '1'
    # MEMORY LEAK PREVENTION: Limit threads for worker processes
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1" 
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    warnings.filterwarnings("ignore")

# Ora importa TensorFlow con le configurazioni applicate
import tensorflow as tf
from tensorflow import keras
import pickle
import h5py

# Configurazione warning globale per processi MPI
def setup_mpi_warnings():
    """Configura i warning per essere mostrati solo dal processo master"""
    if not is_master_process():
        # Disabilita TUTTI i warning per i processi non-master
        warnings.filterwarnings("ignore")
        
        # Disabilita completamente i logger TensorFlow
        import logging
        logging.getLogger('tensorflow').setLevel(logging.FATAL)
        logging.getLogger('tensorflow').disabled = True
        
        # Disabilita anche logger di livello inferiore
        tf.get_logger().setLevel('ERROR')
        tf.autograph.set_verbosity(0)
        
    else:
        # Per il processo master, filtra solo alcuni warning specifici troppo verbosi
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
        warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
        warnings.filterwarnings("ignore", message=".*A NumPy version.*")

# Setup automatico warning MPI
setup_mpi_warnings()

def mpi_print(*args, **kwargs):
    """Stampa solo se siamo sul processo master"""
    if is_master_process():
        print(*args, **kwargs)

def mpi_warning(*args, **kwargs):
    """Stampa warning solo se siamo sul processo master"""
    if is_master_process():
        print(*args, **kwargs)

def mpi_error(*args, **kwargs):
    """Stampa errori solo se siamo sul processo master"""
    if is_master_process():
        print(*args, **kwargs)

def require_finite(name: str, arr):
    """Ensure array is float64 and finite; raise if not."""
    arr64 = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr64)):
        bad = np.logical_not(np.isfinite(arr64))
        raise ValueError(f"{name}: found {bad.sum()} non-finite values")
    return arr64

def _path_basename(path: str) -> str:
    return os.path.basename(str(path)).lower()

class NeuralNetworkCorrector:
    def __init__(self, model_path: str, preprocessor_path: str, scaler_path: str, transform_config_path: Optional[str] = None):
        
        # Carica modello Keras
        try:
            self.model = keras.models.load_model(model_path, compile=False)
            mpi_print("Model loaded successfully")
        except Exception as e:
            mpi_warning(f"[AI] standard load_model failed: {e}")
            try:
                self.model = self._load_model_with_manual_keras_fallback(model_path)
                mpi_print("Model loaded successfully (manual .keras fallback)")
            except Exception as fallback_err:
                mpi_print(f"ERROR loading model: {fallback_err}")
                raise
        
        # Carica preprocessor (StandardScaler + OneHotEncoder)
        try:
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
        except Exception as e:
            mpi_print(f"ERROR loading preprocessor: {e}")
            raise

        # Cache feature bookkeeping so we can drive transformers con NumPy puro.
        # Nomi delle feature: k/nut oppure kMean/nutMean.
        default_feature_names = [
            'Ux',
            'Ur',
            'Ut',
            'k',
            'nut',
            'centroid_dist',
            'z_d',
            'ax_d',
            'clusterID',
        ]
        preprocessor_feature_names = [
            str(name) for name in getattr(self.preprocessor, "feature_names_in_", [])
        ]
        self._solver_feature_names = preprocessor_feature_names or default_feature_names
        self._solver_feature_index = {name: idx for idx, name in enumerate(self._solver_feature_names)}
        if 'clusterID' in self._solver_feature_index:
            self._solver_feature_index.setdefault('ID', self._solver_feature_index['clusterID'])
        if 'k' in self._solver_feature_index:
            self._solver_feature_index.setdefault('kMean', self._solver_feature_index['k'])
        if 'nut' in self._solver_feature_index:
            self._solver_feature_index.setdefault('nutMean', self._solver_feature_index['nut'])
        self._transform_specs = []  # [(name, transformer, [indices])]
        self._requires_manual_preprocessing = False
        self._cluster_categories = None
        self._cluster_alignment_logged = False

        if hasattr(self.preprocessor, 'transformers_'):
            for name, transformer, columns in self.preprocessor.transformers_:
                indices, used_strings = self._resolve_transform_columns(columns)
                if used_strings:
                    self._requires_manual_preprocessing = True
                self._transform_specs.append((name, transformer, indices))
                if name == 'cat' and hasattr(transformer, "categories_") and transformer.categories_:
                    cats = transformer.categories_[0]
                    if len(cats):
                        self._cluster_categories = np.asarray(cats, dtype=np.int64)
        else:
            mpi_warning("Preprocessor has no transformers_. Manual preprocessing will be mandatory.")
            self._requires_manual_preprocessing = True
            self._transform_specs = []
        
        # Carica target scaler (StandardScaler o equivalente)
        try:
            with open(scaler_path, "rb") as f:
                self.y_scaler = pickle.load(f)
        except Exception as e:
            mpi_print(f"ERROR loading y_scaler: {e}")
            raise

        if hasattr(self.y_scaler, "n_features_in_"):
            self._n_targets = int(self.y_scaler.n_features_in_)
        elif hasattr(self.y_scaler, "scale_"):
            self._n_targets = int(np.asarray(self.y_scaler.scale_).size)
        else:
            self._n_targets = 1

        target_mode_env = os.environ.get("NNCORR_TARGET_MODE", "").strip().lower()
        if target_mode_env:
            if target_mode_env not in {"delta", "absolute"}:
                raise ValueError("NNCORR_TARGET_MODE deve essere 'delta' o 'absolute'")
            self._target_mode = target_mode_env
        else:
            name_hint = f"{_path_basename(model_path)} {_path_basename(scaler_path)}"
            self._target_mode = (
                "absolute"
                if "absolute" in name_hint or "direct" in name_hint
                else "delta"
            )

        transform_env = os.environ.get("NNCORR_TARGET_TRANSFORM", "").strip().lower()
        if transform_env:
            if transform_env not in {"none", "log"}:
                raise ValueError("NNCORR_TARGET_TRANSFORM deve essere 'none' o 'log'")
            self._target_transform = transform_env
        else:
            self._target_transform = "log" if "log" in name_hint else "none"

        try:
            self._rotor_diameter_m = float(os.environ.get("NNCORR_ROTOR_DIAMETER", "126.0"))
        except ValueError as exc:
            raise ValueError("NNCORR_ROTOR_DIAMETER deve essere un numero valido") from exc
        if self._rotor_diameter_m <= 0.0:
            raise ValueError("NNCORR_ROTOR_DIAMETER deve essere > 0")

        self._geometry_initialized = False
        self._cyl_abs = False
        self._turbine_name: Optional[str] = None
        self._turbines: Dict[str, Dict[str, Any]] = {}
        self._origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._axis_input = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
        self._ref_axis_base = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self._hub_ax_dist = 0.0
        self._z_tower = 0.0
        self._rotor_yaw = 0.0
        self._rotor_tilt = 0.0
        self._rotor_radius = 0.5*self._rotor_diameter_m

        suppress_debug = os.environ.get("NNCORR_SUPPRESS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
        debug_flag = os.environ.get("NNCORR_DEBUG", "").strip().lower()
        self._debug_enabled = False if suppress_debug else debug_flag in {"1", "true", "yes", "on"}
        try:
            self._debug_limit = max(int(os.environ.get("NNCORR_DEBUG_STEPS", "3")), 0)
        except ValueError:
            self._debug_limit = 3
        self._debug_call_count = 0
        self._feature_debug_done = False
        delta_log_flag = os.environ.get("NNCORR_LOG_DELTA", "1").strip().lower()
        self._delta_log_enabled = delta_log_flag not in {"0", "false", "no", "off"}
        try:
            self._nut_floor_for_log = float(os.environ.get("NNCORR_NUT_FLOOR", "1.5e-5"))
        except ValueError:
            self._nut_floor_for_log = 1.5e-5

        mpi_print(
            f"[AI] Feature order solver->preprocessor: {self._solver_feature_names} "
            f"(rotor diameter={self._rotor_diameter_m} m, "
            f"target_mode={self._target_mode}, target_transform={self._target_transform})"
        )

    def _class_to_checkpoint_base(self, class_name: str) -> str:
        """Convert Keras class name (Dense, LayerNormalization, ...) to checkpoint base name."""
        return re.sub(r"(?<!^)(?=[A-Z])", "_", class_name).lower()

    def _sorted_group_names(self, all_names, base: str):
        """Sort checkpoint group names by numeric suffix (base, base_1, base_2, ...)."""
        pattern = re.compile(rf"^{re.escape(base)}(?:_(\d+))?$")
        indexed = []
        for name in all_names:
            match = pattern.match(name)
            if not match:
                continue
            suffix = match.group(1)
            idx = int(suffix) if suffix is not None else 0
            indexed.append((idx, name))
        indexed.sort(key=lambda item: item[0])
        return [name for _, name in indexed]

    def _array_stats_line(self, label: str, values, prefix: str = "[AI DELTA]") -> str:
        flat = np.asarray(values, dtype=np.float64).ravel()
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return f"{prefix} {label}: no finite values"

        p0, p1, p5, p50, p95, p99, p100 = np.percentile(
            finite, [0, 1, 5, 50, 95, 99, 100]
        )
        return (
            f"{prefix} {label}: min={p0:.6g}, p1={p1:.6g}, p5={p5:.6g}, "
            f"p50={p50:.6g}, p95={p95:.6g}, p99={p99:.6g}, "
            f"max={p100:.6g}, mean={finite.mean():.6g}"
        )

    def _ensure_geometry_initialized(self) -> None:
        """Lazy setup for no-clustering models called with cartesian fields."""
        if self._geometry_initialized:
            return

        self._load_case_config(os.getcwd())
        self._setup_rotated_frame()
        self._geometry_initialized = True

    def _load_case_config(self, case_dir: str) -> None:
        """Read fvOptions/Trim to match the cylindrical frame used at training time."""
        fv_path = os.path.join(case_dir, "system", "fvOptions")
        if not os.path.exists(fv_path):
            mpi_warning(f"fvOptions non trovato in {fv_path}; uso default per geometria")
        else:
            current_turbine: Optional[str] = None
            try:
                with open(fv_path) as f:
                    for raw in f:
                        line = raw.strip()
                        if not line or line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                            continue
                        if line.startswith("turbine"):
                            current_turbine = line.split()[0]
                            self._turbines[current_turbine] = {}
                        if not current_turbine:
                            continue
                        if "rotorRadius" in line:
                            self._turbines[current_turbine]["rotorRadius"] = float(line.split()[1].rstrip(";"))
                        elif "origin" in line:
                            vec = [float(x) for x in line.split("(")[1].split(")")[0].replace(";", "").split()]
                            self._turbines[current_turbine]["origin"] = vec
                        elif "axis" in line:
                            vec = [float(x) for x in line.split("(")[1].split(")")[0].replace(";", "").split()]
                            self._turbines[current_turbine]["axis"] = vec
            except Exception as exc:
                mpi_warning(f"Errore leggendo fvOptions: {exc}; uso default")

        env_turbine = os.environ.get("NNTURBINE", "").strip()
        if env_turbine and env_turbine in self._turbines:
            self._turbine_name = env_turbine
        elif self._turbines:
            self._turbine_name = sorted(self._turbines.keys())[0]

        if self._turbine_name:
            t = self._turbines[self._turbine_name]
            self._origin = np.asarray(t.get("origin", self._origin), dtype=np.float64)
            self._axis_input = np.asarray(t.get("axis", self._axis_input), dtype=np.float64)
            self._rotor_radius = float(t.get("rotorRadius", self._rotor_radius))

        if self._turbine_name:
            trim_path = os.path.join(case_dir, "system", f"Trim.{self._turbine_name}")
            if os.path.exists(trim_path):
                try:
                    with open(trim_path) as f:
                        for raw in f:
                            line = raw.strip()
                            # chiave = primo token (non sottostringa: 'rotorYaw_phase'
                            # non deve sovrascrivere 'rotorYaw')
                            key = line.split()[0] if line.split() else ""
                            if key == "towerHeight":
                                self._z_tower = float(line.split()[1].rstrip(";"))
                            elif key == "hubAxDist":
                                self._hub_ax_dist = -float(line.split()[1].rstrip(";"))
                            elif key == "rotorYaw":
                                self._rotor_yaw = float(line.split()[1].rstrip(";"))
                            elif key == "rotorTilt":
                                self._rotor_tilt = float(line.split()[1].rstrip(";"))
                except Exception as exc:
                    mpi_warning(f"Errore leggendo {trim_path}: {exc}; uso default")

        cyl_abs_env = os.environ.get("CYL_ABS", "").strip().lower()
        if cyl_abs_env in {"1", "true", "yes", "on"}:
            self._cyl_abs = True

    def _setup_rotated_frame(self) -> None:
        rot_yaw = np.array(
            [
                [np.cos(self._rotor_yaw), -np.sin(self._rotor_yaw), 0.0],
                [np.sin(self._rotor_yaw), np.cos(self._rotor_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rot_tilt = np.array(
            [
                [np.cos(self._rotor_tilt), 0.0, np.sin(self._rotor_tilt)],
                [0.0, 1.0, 0.0],
                [-np.sin(self._rotor_tilt), 0.0, np.cos(self._rotor_tilt)],
            ],
            dtype=np.float64,
        )
        rotation_matrix = rot_yaw @ rot_tilt

        axis_dir = -rotation_matrix @ self._axis_input
        ref_axis = rotation_matrix @ self._ref_axis_base

        axis_norm = np.linalg.norm(axis_dir)
        ref_norm = np.linalg.norm(ref_axis)
        if axis_norm == 0.0 or ref_norm == 0.0:
            raise ValueError("Axis/refAxis direction has zero norm after rotation")

        self._axis_dir = axis_dir / axis_norm
        self._ref_axis = ref_axis / ref_norm
        # Centro del frame nel mozzo come nel training (sim_clustering,
        # utils.points_CART2CYL): punto - (origin + hubAxDist, 0, towerHeight).
        hub = np.array(
            [self._origin[0] + self._hub_ax_dist, self._origin[1], self._origin[2] + self._z_tower],
            dtype=np.float64,
        )
        self._point_shift = -hub

        mpi_print("[AI GEOMETRY] setup")
        if self._turbine_name:
            mpi_print(f"[AI GEOMETRY]   turbine: {self._turbine_name}")
        mpi_print(f"[AI GEOMETRY]   origin: {self._origin}")
        mpi_print(f"[AI GEOMETRY]   rotor radius: {self._rotor_radius:.2f} m")
        mpi_print(f"[AI GEOMETRY]   normalization diameter: {self._rotor_diameter_m:.2f} m")
        mpi_print(f"[AI GEOMETRY]   axis direction: {self._axis_dir}")
        mpi_print(f"[AI GEOMETRY]   point shift: {self._point_shift}")

    def compute_cylindrical_velocity(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert cartesian velocity/position arrays to the training cylindrical frame."""
        self._ensure_geometry_initialized()

        ux = require_finite("ux", ux)
        uy = require_finite("uy", uy)
        uz = require_finite("uz", uz)
        x = require_finite("x", x)
        y = require_finite("y", y)
        z = require_finite("z", z)

        points = np.vstack((x, y, z)).T + self._point_shift
        axis_projection = points @ self._axis_dir
        projections = points - np.outer(axis_projection, self._axis_dir)
        radial_distance = np.sqrt(np.sum(projections ** 2, axis=1))

        cross_vals = np.cross(self._ref_axis, projections)
        num = np.einsum("ij,j->i", cross_vals, self._axis_dir)
        den = np.einsum("ij,j->i", projections, self._ref_axis)
        theta = np.arctan2(num, den)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        vectors = np.vstack((ux, uy, uz)).T
        u_axial = vectors @ self._axis_dir
        if self._cyl_abs:
            u_radial = np.abs(cos_theta * uy + sin_theta * uz)
            u_tangential = np.abs(-sin_theta * uy + cos_theta * uz)
        else:
            u_radial = cos_theta * uy + sin_theta * uz
            u_tangential = -sin_theta * uy + cos_theta * uz

        return (
            require_finite("u_axial", u_axial),
            require_finite("u_radial", u_radial),
            require_finite("u_tangential", u_tangential),
            require_finite("axis_projection", axis_projection),
            require_finite("radial_distance", radial_distance),
            require_finite("theta", theta),
        )

    def _load_model_with_manual_keras_fallback(self, model_path: str):
        """
        Robust loader for .keras archives with inconsistent weight naming.
        Rebuild model from config.json and assign variables from model.weights.h5 by
        class-based checkpoint groups and strict shape checks.
        """
        if not model_path.endswith(".keras"):
            raise ValueError(f"Manual fallback supports only .keras files, got '{model_path}'")

        with zipfile.ZipFile(model_path, "r") as archive:
            names = set(archive.namelist())
            if "config.json" not in names or "model.weights.h5" not in names:
                raise ValueError(
                    f"Invalid .keras archive '{model_path}': missing config.json/model.weights.h5"
                )

            config = json.loads(archive.read("config.json"))
            model = keras.models.model_from_json(json.dumps(config))

            with tempfile.TemporaryDirectory(prefix="nncorr_keras_") as tmpdir:
                weights_path = archive.extract("model.weights.h5", path=tmpdir)

                with h5py.File(weights_path, "r") as h5f:
                    if "_layer_checkpoint_dependencies" not in h5f:
                        raise ValueError("model.weights.h5 missing '_layer_checkpoint_dependencies'")

                    layer_deps = h5f["_layer_checkpoint_dependencies"]
                    all_group_names = list(layer_deps.keys())

                    base_names = {
                        self._class_to_checkpoint_base(layer.__class__.__name__)
                        for layer in model.layers
                    }
                    groups_by_base = {
                        base: self._sorted_group_names(all_group_names, base)
                        for base in base_names
                    }
                    positions = {base: 0 for base in base_names}

                    for layer in model.layers:
                        base = self._class_to_checkpoint_base(layer.__class__.__name__)
                        candidates = groups_by_base.get(base, [])
                        pos = positions.get(base, 0)
                        if pos >= len(candidates):
                            raise ValueError(
                                f"Missing checkpoint group for layer '{layer.name}' "
                                f"(class={layer.__class__.__name__}, base={base}, index={pos})"
                            )

                        group_name = candidates[pos]
                        positions[base] = pos + 1

                        vars_group = layer_deps[group_name].get("vars")
                        arrays = []
                        if vars_group is not None:
                            var_keys = sorted(vars_group.keys(), key=lambda key: int(key))
                            arrays = [np.asarray(vars_group[key]) for key in var_keys]

                        expected_shapes = [tuple(var.shape.as_list()) for var in layer.weights]
                        found_shapes = [tuple(arr.shape) for arr in arrays]
                        if expected_shapes != found_shapes:
                            raise ValueError(
                                f"Weight shape mismatch for layer '{layer.name}' from group '{group_name}': "
                                f"expected={expected_shapes}, found={found_shapes}"
                            )

                        if arrays:
                            layer.set_weights(arrays)

        return model
    
    def _align_cluster_ids(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Allinea gli ID cluster al dominio dell'encoder (es. 1..k invece di 0..k-1)."""
        ids = np.asarray(cluster_ids, dtype=np.int64)

        if ids.size == 0 or self._cluster_categories is None:
            return ids

        expected_min = int(self._cluster_categories.min())
        expected_max = int(self._cluster_categories.max())
        obs_min = int(ids.min())
        obs_max = int(ids.max())

        # Diagnostica dettagliata (solo prima volta)
        if not self._cluster_alignment_logged:
            unique_ids = np.unique(ids)
            counts = {int(cid): int(np.sum(ids == cid)) for cid in unique_ids}
            
            mpi_print(f"[AI] ═══════════════════════════════════════════════")
            mpi_print(f"[AI] CLUSTER ALIGNMENT CHECK:")
            mpi_print(f"[AI]   Received from clustering: {obs_min} to {obs_max}")
            mpi_print(f"[AI]   Expected by OneHotEncoder: {expected_min} to {expected_max}")
            mpi_print(f"[AI]   Unique clusters: {len(unique_ids)}")
            mpi_print(f"[AI]   Distribution: {dict(sorted(counts.items()))}")

        # Caso tipico 1: KMeans 0-based, encoder 1-based
        if obs_min == 0 and expected_min == 1 and obs_max <= expected_max - 1:
            shift = expected_min - obs_min
            if not self._cluster_alignment_logged:
                mpi_print(f"[AI]   ✓ Applying shift +{shift} (0-based → 1-based)")
                mpi_print(f"[AI] ═══════════════════════════════════════════════")
                self._cluster_alignment_logged = True
            return ids + shift

        # Caso tipico 2: KMeans 1-based, encoder 0-based
        if obs_min == 1 and expected_min == 0 and obs_max <= expected_max + 1:
            shift = expected_min - obs_min
            if not self._cluster_alignment_logged:
                mpi_print(f"[AI]   ✓ Applying shift {shift} (1-based → 0-based)")
                mpi_print(f"[AI] ═══════════════════════════════════════════════")
                self._cluster_alignment_logged = True
            return ids + shift

        # Caso 3: Range già allineato
        if obs_min == expected_min and obs_max == expected_max:
            if not self._cluster_alignment_logged:
                mpi_print(f"[AI]   ✓ Cluster IDs already aligned")
                mpi_print(f"[AI] ═══════════════════════════════════════════════")
                self._cluster_alignment_logged = True
            return ids

        # ERRORE CRITICO: fuori range (non clippiamo, falliamo!)
        if obs_min < expected_min or obs_max > expected_max:
            error_msg = (
                f"\n{'═'*70}\n"
                f"[AI FATAL] CLUSTER ID MISMATCH!\n"
                f"{'═'*70}\n"
                f"  Received: {obs_min} to {obs_max}\n"
                f"  Expected: {expected_min} to {expected_max}\n"
                f"\n"
                f"  PROBABLE CAUSES:\n"
                f"  1. clustering.py returns 0-based but training was 1-based\n"
                f"     → FIX: Add '+1' in clustering.py line ~113\n"
                f"  2. Different number of clusters than training\n"
                f"     → CHECK: KMeans n_clusters in models/km\n"
                f"{'═'*70}\n"
            )
            raise ValueError(error_msg)

        if not self._cluster_alignment_logged:
            mpi_print(f"[AI]   ✓ Cluster IDs passed through unchanged")
            mpi_print(f"[AI] ═══════════════════════════════════════════════")
            self._cluster_alignment_logged = True

        return ids

    def predict_corrections(self, u_axial: np.ndarray, u_tangential: np.ndarray,
                            u_radial: np.ndarray, k: np.ndarray, nut: np.ndarray,
                            centroid_dist: np.ndarray, cluster_ids: np.ndarray,
                            ax_coord_m: np.ndarray, z_coord_m: np.ndarray) -> tuple:
        """
        Predice nut. Il target del modello (correzione additiva "delta" o
        valore assoluto, eventualmente in spazio log) e' dedotto dal nome dei
        file o impostato con NNCORR_TARGET_MODE / NNCORR_TARGET_TRANSFORM.
        Le feature seguono rigorosamente il training salvato nel preprocessor.
        Sono supportati sia i modelli con clustering
        [Ux, Ur, Ut, k, nut, centroid_dist, z_d, ax_d, clusterID]
        sia quelli senza clustering [Ux, Ur, Ut, k, nut, z_d, ax_d].
        Sono supportati target [nut] e, per compatibilita', [nut, k] o delta.
        """
        fallback_nut = np.asarray(nut, dtype=np.float64)
        fallback_k = np.asarray(k, dtype=np.float64)

        if self._debug_enabled:
            mpi_print("Running neural network prediction...")

        try:
            # Controlli di sicurezza
            u_axial = require_finite("u_axial", u_axial)
            u_tangential = require_finite("u_tangential", u_tangential)
            u_radial = require_finite("u_radial", u_radial)
            k_for_model = require_finite("k", k)
            nut_for_model = require_finite("nut", nut)
            ax_coord_m = require_finite("ax_coord_m", ax_coord_m)
            z_coord_m = require_finite("z_coord_m", z_coord_m)

            z_d = z_coord_m / self._rotor_diameter_m
            ax_d = ax_coord_m / self._rotor_diameter_m

            needs_centroid = 'centroid_dist' in self._solver_feature_index
            if needs_centroid:
                centroid_dist = require_finite("centroid_dist", centroid_dist)
            else:
                centroid_dist = np.zeros_like(k_for_model, dtype=np.float64)

            needs_cluster = (
                'clusterID' in self._solver_feature_index
                or 'ID' in self._solver_feature_index
            )
            if needs_cluster:
                cluster_ids_for_model = self._align_cluster_ids(cluster_ids)
            else:
                cluster_ids_for_model = np.zeros_like(k_for_model, dtype=np.int64)

            if self._debug_enabled and not self._feature_debug_done:
                def _fmt_stats(name, values):
                    flat = np.asarray(values).astype(np.float64).ravel()
                    finite = flat[np.isfinite(flat)]
                    if finite.size == 0:
                        return f"{name}: valori non finiti"
                    return (
                        f"{name}: shape={flat.shape}, min={finite.min():.6g}, "
                        f"max={finite.max():.6g}, mean={finite.mean():.6g}, std={finite.std():.6g}"
                    )

                mpi_print("[AI DEBUG] Statistiche feature input (prima chiamata)")
                mpi_print(_fmt_stats("Ux", u_axial))
                mpi_print(_fmt_stats("Ur", u_radial))
                mpi_print(_fmt_stats("Ut", u_tangential))
                mpi_print(_fmt_stats("k", k_for_model))
                mpi_print(_fmt_stats("nut", nut_for_model))
                if needs_centroid:
                    mpi_print(_fmt_stats("centroid_dist", centroid_dist))
                mpi_print(_fmt_stats("z_d", z_d))
                mpi_print(_fmt_stats("ax_d", ax_d))
                if needs_cluster:
                    ids = np.asarray(cluster_ids_for_model).astype(np.int64).ravel()
                    if ids.size:
                        ids_min = int(ids.min())
                        ids_max = int(ids.max())
                        mpi_print(f"cluster_ids: shape={ids.shape}, range=[{ids_min}, {ids_max}], unique~={np.unique(ids).size}")
                self._feature_debug_done = True

            feature_map = {
                'Ux': u_axial,
                'Ur': u_radial,
                'Ut': u_tangential,
                'k': k_for_model,
                'kMean': k_for_model,
                'nut': nut_for_model,
                'nutMean': nut_for_model,
                'centroid_dist': centroid_dist,
                'z_d': z_d,
                'ax_d': ax_d,
                'clusterID': cluster_ids_for_model.astype(int),
                'ID': cluster_ids_for_model.astype(int),
            }

            missing_features = [
                name for name in self._solver_feature_names
                if name not in feature_map
            ]
            if missing_features:
                raise RuntimeError(
                    f"Feature mancanti per il modello: {missing_features}. "
                    f"Feature disponibili: {sorted(feature_map)}"
                )

            X_for_preprocessing = np.column_stack([
                feature_map[name] for name in self._solver_feature_names
            ])

            if self._requires_manual_preprocessing:
                X_prep = self._apply_manual_preprocessing(X_for_preprocessing)
            else:
                X_prep = self.preprocessor.transform(X_for_preprocessing)
            X_prep = np.asarray(X_prep, dtype=np.float64)

            if isinstance(self.model.inputs, (list, tuple)):
                input_tensors = list(self.model.inputs)
            else:
                input_tensors = [self.model.inputs]

            expected_features = None
            if input_tensors:
                ef = input_tensors[0].shape[1]
                expected_features = int(ef) if ef is not None else None

            if expected_features is not None and X_prep.shape[1] != expected_features:
                raise RuntimeError(
                    f"Feature mismatch: model expects {expected_features} features but input has {X_prep.shape[1]}. "
                    f"Verify feature order {self._solver_feature_names}."
                )

            if len(input_tensors) <= 1:
                model_inputs = X_prep
            else:
                nut_col = nut_for_model.reshape(-1, 1)
                expected_nut = input_tensors[1].shape[1]
                expected_nut = int(expected_nut) if expected_nut is not None else None
                if expected_nut is not None and nut_col.shape[1] != expected_nut:
                    raise RuntimeError(
                        f"Nut input mismatch: model expects nut input shape (*,{expected_nut}) "
                        f"but got (*,{nut_col.shape[1]})"
                    )
                model_inputs = {
                    input_tensors[0].name.split(':')[0]: X_prep,
                    input_tensors[1].name.split(':')[0]: nut_col,
                }

            delta_pred_scaled = self.model.predict(model_inputs, verbose=0, batch_size=1024)
            if isinstance(delta_pred_scaled, (list, tuple)):
                if len(delta_pred_scaled) == 1:
                    delta_pred_scaled = delta_pred_scaled[0]
                else:
                    delta_pred_scaled = np.concatenate(
                        [np.asarray(out).reshape(np.asarray(out).shape[0], -1) for out in delta_pred_scaled],
                        axis=1
                    )
            delta_pred_scaled = np.asarray(delta_pred_scaled, dtype=np.float64)
            if delta_pred_scaled.ndim == 1:
                delta_pred_scaled = delta_pred_scaled.reshape(-1, 1)
            elif delta_pred_scaled.ndim > 2:
                delta_pred_scaled = delta_pred_scaled.reshape(delta_pred_scaled.shape[0], -1)

            if delta_pred_scaled.shape[1] != self._n_targets:
                raise RuntimeError(
                    f"Target mismatch: model predicted {delta_pred_scaled.shape[1]} target(s) "
                    f"but y_scaler expects {self._n_targets}"
                )

            target_phys = self.y_scaler.inverse_transform(delta_pred_scaled)
            if self._target_transform == "log":
                target_phys = np.exp(np.clip(target_phys, -745.0, 709.0))

            target_nut = target_phys[:, 0].ravel()
            if self._target_mode == "delta":
                delta_nut = target_nut
                nut_corrected = nut_for_model + delta_nut
            else:
                nut_corrected = target_nut
                delta_nut = nut_corrected - nut_for_model

            # posizioni 2-3 della tupla: k in ingresso, invariato, e incremento nullo
            k_input = k_for_model
            k_increment = np.zeros_like(k_for_model)

            if self._delta_log_enabled:
                corrected_flat = np.asarray(nut_corrected, dtype=np.float64).ravel()
                finite_corrected = corrected_flat[np.isfinite(corrected_flat)]
                if finite_corrected.size:
                    below_zero_pct = 100.0 * np.mean(finite_corrected < 0.0)
                    below_floor_pct = 100.0 * np.mean(finite_corrected <= self._nut_floor_for_log)
                else:
                    below_zero_pct = 0.0
                    below_floor_pct = 0.0

                mpi_print(self._array_stats_line("nut_input", nut_for_model))
                mpi_print(self._array_stats_line("delta_nut", delta_nut))
                mpi_print(
                    self._array_stats_line("nut_corrected_raw", nut_corrected)
                    + f", below0={below_zero_pct:.2f}%, belowFloor={below_floor_pct:.2f}%"
                )

            if self._debug_enabled and self._debug_call_count < self._debug_limit:
                self._debug_call_count += 1

                def _summarize(label, arr):
                    flat = np.asarray(arr, dtype=np.float64).ravel()
                    finite = flat[np.isfinite(flat)]
                    if finite.size == 0:
                        mpi_print(f"[AI DEBUG] {label}: nessun valore finito")
                        return
                    pcts = np.percentile(finite, [0, 5, 50, 95, 100])
                    mpi_print(
                        f"[AI DEBUG] {label}: min={finite.min():.6g}, max={finite.max():.6g}, "
                        f"mean={finite.mean():.6g}, p5={pcts[1]:.6g}, p95={pcts[3]:.6g}"
                    )

                _summarize("delta_nut", delta_nut)
                _summarize("nut_corrected", nut_corrected)

            return nut_corrected, delta_nut, k_input, k_increment

        except Exception as e:
            mpi_error(f"ERROR in neural network prediction: {e}")
            import traceback
            traceback.print_exc()
            return (
                fallback_nut,
                np.zeros_like(fallback_nut),
                fallback_k,
                np.zeros_like(fallback_k),
            )

    def predict_corrections_from_cartesian(self, ux: np.ndarray, uy: np.ndarray,
                                           uz: np.ndarray, x: np.ndarray,
                                           y: np.ndarray, z: np.ndarray,
                                           k: np.ndarray, nut: np.ndarray) -> tuple:
        """
        Entry point for no-clustering solvers.
        Converts cartesian fields to cylindrical training features, then calls
        predict_corrections with dummy clustering arrays that are ignored by
        no-clustering preprocessors.
        """
        (
            u_axial,
            u_radial,
            u_tangential,
            axial_coord,
            radial_distance,
            theta,
        ) = self.compute_cylindrical_velocity(ux, uy, uz, x, y, z)

        k_for_shape = np.asarray(k, dtype=np.float64)
        dummy_centroid = np.zeros_like(k_for_shape, dtype=np.float64)
        dummy_cluster = np.zeros_like(k_for_shape, dtype=np.int64)

        result = self.predict_corrections(
            u_axial,
            u_tangential,
            u_radial,
            k,
            nut,
            dummy_centroid,
            dummy_cluster,
            axial_coord,
            z,
        )

        return (
            result[0],
            result[1],
            result[2],
            result[3],
            u_axial,
            u_radial,
            u_tangential,
            axial_coord,
            radial_distance,
            theta,
        )

    def predict_nut_correction(self, *args, **kwargs) -> tuple:
        """Backward-compatible wrapper for older C++ call sites."""
        return self.predict_corrections(*args, **kwargs)

    def _resolve_transform_columns(self, columns):
        """Convert column spec to numeric indices; segnala se usava stringhe."""
        uses_strings = False
        feature_lookup = self._solver_feature_index

        def resolve_single(entry):
            nonlocal uses_strings

            if isinstance(entry, str):
                uses_strings = True
                idx = feature_lookup.get(entry)
                if idx is not None:
                    return idx
                mpi_warning(f"Preprocessor column '{entry}' non riconosciuta: verrà ignorata")
                return None

            if isinstance(entry, (np.integer, int)):
                return int(entry)

            return None

        if isinstance(columns, slice):
            start = columns.start or 0
            stop = columns.stop if columns.stop is not None else len(self._solver_feature_names)
            step = columns.step or 1
            indices = list(range(start, stop, step))
            return indices, uses_strings

        if isinstance(columns, (list, tuple, np.ndarray)):
            indices = []
            for entry in columns:
                if isinstance(entry, slice):
                    sub_indices, sub_uses_strings = self._resolve_transform_columns(entry)
                    uses_strings = uses_strings or sub_uses_strings
                    indices.extend(sub_indices)
                else:
                    idx = resolve_single(entry)
                    if idx is not None:
                        indices.append(idx)
            return indices, uses_strings

        idx = resolve_single(columns)
        indices = [] if idx is None else [idx]
        return indices, uses_strings

    def _apply_manual_preprocessing(self, X_for_preprocessing):
        """Riproduce ColumnTransformer usando solo NumPy e transformer fitted."""
        if not self._transform_specs:
            raise Exception("No transformer specs available for manual preprocessing")

        X_prep_parts = []

        for name, transformer, indices in self._transform_specs:
            if name == 'remainder' or not indices:
                continue

            X_subset = X_for_preprocessing[:, indices]

            try:
                X_transformed = transformer.transform(X_subset)
                if hasattr(X_transformed, "toarray"):
                    X_transformed = X_transformed.toarray()
                X_prep_parts.append(X_transformed)
            except Exception as trans_err:
                mpi_error(f"ERROR in transformer '{name}': {trans_err}")
                X_prep_parts.append(X_subset)

        if not X_prep_parts:
            raise Exception("All preprocessing transformers failed")

        return np.concatenate(X_prep_parts, axis=1)
