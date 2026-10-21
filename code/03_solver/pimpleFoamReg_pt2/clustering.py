import numpy as np
import joblib
import os
from typing import Tuple, List, Optional, Dict, Any
from sklearn.cluster import KMeans
import warnings

# Supporto MPI per messaggi puliti
def get_mpi_rank():
    """Ottieni il rank MPI corrente. Ritorna 0 se non in ambiente MPI."""
    try:
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

# Configurazione warning globale per processi MPI
def setup_mpi_warnings():
    """Configura i warning per essere mostrati solo dal processo master"""
    if not is_master_process():
        # Disabilita TUTTI i warning per i processi non-master
        warnings.filterwarnings("ignore")
        
        # Disabilita anche logger sklearn specifici
        import logging
        logging.getLogger('sklearn').setLevel(logging.ERROR)
        
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

def require_finite(name: str, arr: np.ndarray) -> np.ndarray:
    """Ensure array is float64 and finite; raise if not."""
    arr64 = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(arr64)):
        bad = np.logical_not(np.isfinite(arr64))
        raise ValueError(f"{name}: found {bad.sum()} non-finite values")
    return arr64

class ClusteringProcessor:
    def __init__(self, model_path: str, scaler_path: str):
        mpi_print(f"Loading clustering components...")
        self.kmeans_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self._feature_order: List[str] = [str(name) for name in getattr(self.scaler, "feature_names_in_", [])]
        if not self._feature_order:
            raise ValueError("Scaler non contiene 'feature_names_in_'")

        # --- Configurazione geometria/turbina (meno hardcoded possibile) ---
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
        self._rotor_radius = 63.0  # Default 5MW turbine, verrà aggiornato da fvOptions

        self._load_case_config(os.getcwd())
        self._setup_rotated_frame()

        # Garantisce coerenza con il backend scikit-learn (compilato in float64)
        self.kmeans_model.cluster_centers_ = np.require(
            np.asarray(self.kmeans_model.cluster_centers_, dtype=np.float64),
            dtype=np.float64,
            requirements=("C", "A"),
        )
        if hasattr(self.kmeans_model, "init") and isinstance(self.kmeans_model.init, np.ndarray):
            self.kmeans_model.init = np.require(
                np.asarray(self.kmeans_model.init, dtype=np.float64),
                dtype=np.float64,
                requirements=("C", "A"),
            )

        mpi_print(f"✓ Clustering model loaded (features: {self._feature_order})")

        # ------------------------------------------------------------------ #
        # RIORDINO DEI CLUSTER (identico al training in sim_clustering).
        # Il training fa: label KMeans grezze -> (+1) -> reordering() che
        # riordina gli ID per MEDIA della variabile 'clusterReorderMethod' in
        # ordine ascending/descending e riassegna new_ID = 1..k.
        # Config di training usata per questi modelli:
        #   clusterReorderMethod = "Ux"  ;  ascendingOrder = False (discendente)
        # => Ux medio piu' alto  ==  flusso indisturbato  ==  new_ID 1.
        #
        # Lo scaler dei cluster e' MinMax (affine, monotono) e i centroidi
        # KMeans sono la media (in spazio scalato) dei punti assegnati; quindi
        # l'ordinamento dei cluster per media di 'Ux' coincide ESATTAMENTE con
        # l'ordinamento dei centroidi lungo 'Ux'. Ricostruiamo la mappa
        # raw->new_ID dai soli centroidi del modello km salvato: e' una
        # proprieta' FISSA del modello, NON va ricalcolata per caso/timestep.
        # ------------------------------------------------------------------ #
        n_clusters = int(self.kmeans_model.n_clusters)
        reorder_var = os.environ.get("NNCLUSTER_REORDER_VAR", "Ux").strip()
        asc_env = os.environ.get("NNCLUSTER_REORDER_ASCENDING", "false").strip().lower()
        reorder_ascending = asc_env in {"1", "true", "yes", "on"}
        if reorder_var in self._feature_order:
            var_idx = self._feature_order.index(reorder_var)
            center_var = np.asarray(
                self.kmeans_model.cluster_centers_, dtype=np.float64
            )[:, var_idx]
            order = np.argsort(center_var, kind="stable")        # ascending
            if not reorder_ascending:
                order = order[::-1]                               # descending
            # order[rank] = label grezza KMeans; new_ID (1-based) = rank + 1
            new_id_for_raw = np.empty(n_clusters, dtype=np.int32)
            new_id_for_raw[order] = np.arange(1, n_clusters + 1, dtype=np.int32)
            self._cluster_remap = new_id_for_raw
            mpi_print(
                f"[CLUSTERING] Riordino cluster per media '{reorder_var}' "
                f"({'ascending' if reorder_ascending else 'descending'}): "
                f"raw KMeans 0-based -> new_ID 1-based = {new_id_for_raw.tolist()} "
                f"(flusso indisturbato = ID 1, scia profonda = ID {n_clusters})"
            )
        else:
            self._cluster_remap = np.arange(1, n_clusters + 1, dtype=np.int32)
            mpi_warning(
                f"[CLUSTERING] Variabile di riordino '{reorder_var}' non presente "
                f"nelle feature {self._feature_order}: uso ID 1-based nell'ordine "
                f"grezzo KMeans (NESSUN riordino fisico!)"
            )

    # ------------------------------------------------------------------ #
    # Configurazione geometria/turbina
    # ------------------------------------------------------------------ #
    def _load_case_config(self, case_dir: str) -> None:
        """Legge fvOptions/Trim per ottenere parametri turbina e frame cilindrico."""
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
                        elif "freeStreamVelocity" in line:
                            vec = [float(x) for x in line.split("(")[1].split(")")[0].replace(";", "").split()]
                            self._turbines[current_turbine]["freeStreamVelocity"] = vec
                        elif "tipSpeedRatio" in line:
                            self._turbines[current_turbine]["tipSpeedRatio"] = float(line.split()[1].rstrip(";"))
            except Exception as exc:
                mpi_warning(f"Errore leggendo fvOptions: {exc}; uso default")

        # Seleziona turbina: env NNTURBINE altrimenti la prima
        env_turbine = os.environ.get("NNTURBINE", "").strip()
        if env_turbine and env_turbine in self._turbines:
            self._turbine_name = env_turbine
        elif self._turbines:
            self._turbine_name = sorted(self._turbines.keys())[0]

        if self._turbine_name:
            t = self._turbines[self._turbine_name]
            self._origin = np.asarray(t.get("origin", self._origin), dtype=np.float64)
            self._axis_input = np.asarray(t.get("axis", self._axis_input), dtype=np.float64)
            self._rotor_radius = t.get("rotorRadius", self._rotor_radius)

        # Lettura Trim.<turbina> se presente
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
                    mpi_warning(f"Errore leggendo {trim_path}: {exc}; uso default per yaw/tilt/hubAxDist/z_tower")

        cyl_abs_env = os.environ.get("CYL_ABS", "").strip().lower()
        if cyl_abs_env in {"1", "true", "yes", "on"}:
            self._cyl_abs = True

    def _setup_rotated_frame(self) -> None:
        """Imposta assi ruotati (yaw+tilt) coerenti con il training."""
        # Rotazioni yaw (z) poi tilt (y)
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
        
        # Diagnostica geometria
        mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")
        mpi_print(f"[CLUSTERING] GEOMETRY SETUP:")
        if self._turbine_name:
            mpi_print(f"[CLUSTERING]   Turbine: {self._turbine_name}")
        mpi_print(f"[CLUSTERING]   Origin: {self._origin}")
        mpi_print(f"[CLUSTERING]   Rotor radius: {self._rotor_radius:.2f} m (diameter: {2*self._rotor_radius:.2f} m)")
        mpi_print(f"[CLUSTERING]   Axis input: {self._axis_input}")
        mpi_print(f"[CLUSTERING]   Yaw: {np.degrees(self._rotor_yaw):.2f}°, Tilt: {np.degrees(self._rotor_tilt):.2f}°")
        mpi_print(f"[CLUSTERING]   Axis direction (final): {self._axis_dir}")
        mpi_print(f"[CLUSTERING]   Ref axis (theta=0): {self._ref_axis}")
        mpi_print(f"[CLUSTERING]   Point shift: {self._point_shift}")
        dot_product = abs(self._axis_dir @ self._ref_axis)
        mpi_print(f"[CLUSTERING]   Orthogonality check: dot={dot_product:.2e} (should be ~0)")
        if dot_product > 1e-3:
            mpi_warning(f"[CLUSTERING] WARNING: ref_axis non ortogonale ad axis_dir!")
        mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")

    def compute_cylindrical_velocity(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Converte velocità nel frame cilindrico e calcola ax, rho, theta."""

        ux = require_finite("ux", ux)
        uy = require_finite("uy", uy)
        uz = require_finite("uz", uz)
        x = require_finite("x", x)
        y = require_finite("y", y)
        z = require_finite("z", z)

        # Traslazione verso il frame cilindrico usato in training
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

        u_axial = require_finite("u_axial", u_axial)
        u_radial = require_finite("u_radial", u_radial)
        u_tangential = require_finite("u_tangential", u_tangential)

        axis_projection = require_finite("axis_projection", axis_projection)
        radial_distance = require_finite("radial_distance", radial_distance)
        theta = require_finite("theta", theta)

        return (
            u_axial,
            u_radial,
            u_tangential,
            axis_projection,
            radial_distance,
            theta,
        )

    def perform_clustering_with_distances(
        self,
        ux: np.ndarray,
        uy: np.ndarray,
        uz: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        k: np.ndarray,
        nut: np.ndarray,
        vort_cart_x: Optional[np.ndarray] = None,
        vort_cart_y: Optional[np.ndarray] = None,
        vort_cart_z: Optional[np.ndarray] = None,
    ) -> tuple:
        """Restituisce velocità cilindriche, ID di cluster, distanza/coordinate cilindriche."""

        k = require_finite("k", k)
        nut = require_finite("nut", nut)

        (
            u_axial,
            u_radial,
            u_tangential,
            axial_coord,
            radial_distance,
            theta,
        ) = self.compute_cylindrical_velocity(ux, uy, uz, x, y, z)

        needs_vorticity = any(
            name in self._feature_order
            for name in ("vort_x", "vort_t", "vort_r")
        )
        vort_axial = None
        vort_tangential = None
        vort_radial = None
        if needs_vorticity:
            if vort_cart_x is None or vort_cart_y is None or vort_cart_z is None:
                raise ValueError(
                    "Il modello di clustering richiede vort_x/vort_t/vort_r, "
                    "ma il solver non ha passato la vorticita'"
                )
            vort_cart_x = require_finite("vort_cart_x", vort_cart_x)
            vort_cart_y = require_finite("vort_cart_y", vort_cart_y)
            vort_cart_z = require_finite("vort_cart_z", vort_cart_z)

            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            vort_vectors = np.vstack((vort_cart_x, vort_cart_y, vort_cart_z)).T
            vort_axial = vort_vectors @ self._axis_dir
            if self._cyl_abs:
                vort_radial = np.abs(cos_theta * vort_cart_y + sin_theta * vort_cart_z)
                vort_tangential = np.abs(-sin_theta * vort_cart_y + cos_theta * vort_cart_z)
            else:
                vort_radial = cos_theta * vort_cart_y + sin_theta * vort_cart_z
                vort_tangential = -sin_theta * vort_cart_y + cos_theta * vort_cart_z

            vort_axial = require_finite("vort_axial", vort_axial)
            vort_tangential = require_finite("vort_tangential", vort_tangential)
            vort_radial = require_finite("vort_radial", vort_radial)

        # Log range coordinate (solo prima volta)
        # NOTA: Il modello usa coordinate in METRI (non normalizzate in diametri)
        if not hasattr(self, '_coord_range_logged'):
            self._coord_range_logged = True
            mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")
            mpi_print(f"[CLUSTERING] COORDINATE RANGES (in meters):")
            mpi_print(f"[CLUSTERING]   ax  range: [{axial_coord.min():.1f}, {axial_coord.max():.1f}] m")
            mpi_print(f"[CLUSTERING]   rho range: [{radial_distance.min():.1f}, {radial_distance.max():.1f}] m")
            mpi_print(f"[CLUSTERING]   z   range: [{z.min():.1f}, {z.max():.1f}] m")
            mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")
        
        feature_map = {
            "Ux": u_axial,
            "Ur": u_radial,
            "Ut": u_tangential,
            "Uy": uy,
            "Uz": uz,
            "x": x,
            "y": y,
            "z": z,                    # z in METRI
            "z_original": z,           # Alias per compatibilità con nuovo modello
            "k": k,
            "nut": nut,
            "u_axial": u_axial,
            "u_radial": u_radial,
            "u_tangential": u_tangential,
            "vort_x": vort_axial,
            "vort_t": vort_tangential,
            "vort_r": vort_radial,
            "ax": axial_coord,         # ax in METRI
            "rho": radial_distance,    # rho in METRI
            "theta": theta,
        }

        ordered_columns: List[np.ndarray] = []
        for name in self._feature_order:
            array = feature_map.get(name)
            if array is None:
                raise ValueError(f"Feature '{name}' mancante per il clustering")
            ordered_columns.append(np.asarray(array, dtype=np.float64))

        features = np.require(
            np.column_stack(ordered_columns),
            dtype=np.float64,
            requirements=("C", "A"),
        )
        features_scaled = np.require(
            self.scaler.transform(features),
            dtype=np.float64,
            requirements=("C", "A"),
        )
        features_scaled = require_finite("features_scaled", features_scaled)

        if features_scaled.dtype != np.float64:
            raise ValueError(
                f"Unexpected dtype from scaler/stacking: {features_scaled.dtype}"
            )

        cluster_ids = np.asarray(self.kmeans_model.predict(features_scaled), dtype=np.int32)
        centers = self.kmeans_model.cluster_centers_
        centroid_dist = np.linalg.norm(features_scaled - centers[cluster_ids], axis=1)

        cluster_ids_out = self._cluster_remap[cluster_ids]
        centroid_dist = np.asarray(centroid_dist, dtype=np.float64)

        # Diagnostica distribuzione cluster
        if is_master_process():
            unique_ids = np.unique(cluster_ids_out)
            counts = {int(cid): int(np.sum(cluster_ids_out == cid)) for cid in unique_ids}
            total_points = len(cluster_ids_out)
            
            mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")
            mpi_print(f"[CLUSTERING] CLUSTER DISTRIBUTION:")
            for cid in sorted(counts.keys()):
                percentage = 100.0 * counts[cid] / total_points
                mpi_print(f"[CLUSTERING]   Cluster {cid:2d}: {counts[cid]:6d} cells ({percentage:5.2f}%)")
            mpi_print(f"[CLUSTERING] Cluster ID range: {int(unique_ids.min())}-{int(unique_ids.max())}")
            mpi_print(f"[CLUSTERING] ═══════════════════════════════════════")
            
            # Warning se troppo sbilanciato
            dominant_cluster = max(counts, key=counts.get)
            dominant_ratio = counts[dominant_cluster] / total_points
            
            if dominant_ratio > 0.80:
                mpi_warning(f"[CLUSTERING] ⚠️  CLUSTER IMBALANCE: {dominant_ratio*100:.1f}% in cluster {dominant_cluster}!")
                mpi_warning(f"[CLUSTERING] Possibili cause:")
                mpi_warning(f"[CLUSTERING]   - Feature scaling errato")
                mpi_warning(f"[CLUSTERING]   - Conversione coordinate cilindriche sbagliata")
                mpi_warning(f"[CLUSTERING]   - Asse/origine turbina errati")
            
            if len(unique_ids) < self.kmeans_model.n_clusters / 2:
                mpi_warning(f"[CLUSTERING] ⚠️  Solo {len(unique_ids)}/{self.kmeans_model.n_clusters} cluster utilizzati!")

        return (
            u_axial,
            u_radial,
            u_tangential,
            cluster_ids_out,
            centroid_dist,
            axial_coord,
            radial_distance,
            theta,
        )
