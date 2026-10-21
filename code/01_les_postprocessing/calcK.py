import pyvista as pv

path = "ref_les\\les_mesh.vtp"
mesh = pv.read(path)
turb_field = mesh["turbulenceProperties:R"]
Uprime2Mean = mesh["UPrime2Mean"]
k = 0.5 * (Uprime2Mean[:, 0] + Uprime2Mean[:, 1] + Uprime2Mean[:, 2])+0.5*(turb_field[:,0]+turb_field[:,1]+turb_field[:,2])
mesh["k"] = k
k_sgs = 0.5*(turb_field[:,0]+turb_field[:,1]+turb_field[:,2])
mesh["k_sgs"] = k_sgs
k_res = k - k_sgs
mesh["k_res"] = k_res
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

# Calcola omega evitando divisione per zero
nutEq = mesh["nutEq"].copy()
mask_valid = nutEq > 1e-10
omega = np.zeros_like(nutEq)
omega[mask_valid] = mesh["k"][mask_valid] / nutEq[mask_valid]

# Rimuovi outlier (valori troppo alti)
percentile_95 = np.percentile(omega[mask_valid], 95)
omega[omega > percentile_95] = percentile_95

# RBF Interpolation per i punti non validi - molto più smooth
if np.sum(~mask_valid) > 0:
    points = np.array(mesh.points)
    valid_points = points[mask_valid]
    invalid_points = points[~mask_valid]
    valid_omega = omega[mask_valid]
    
    # Sottocampiona i punti validi se sono troppi (per velocità)
    max_points = 10000
    if len(valid_points) > max_points:
        idx_sample = np.random.choice(len(valid_points), max_points, replace=False)
        valid_points_sample = valid_points[idx_sample]
        valid_omega_sample = valid_omega[idx_sample]
    else:
        valid_points_sample = valid_points
        valid_omega_sample = valid_omega
    
    # RBF con kernel Gaussiano - risultato molto smooth
    rbf = RBFInterpolator(valid_points_sample, valid_omega_sample, 
                          kernel='thin_plate_spline', smoothing=1.0)
    omega[~mask_valid] = rbf(invalid_points)

# Smoothing finale: media mobile su tutto il campo
def smooth_all_field(points, field, k_neighbors=10):
    tree = cKDTree(points)
    distances, idx = tree.query(points, k=k_neighbors)
    distances = np.maximum(distances, 1e-10)
    weights = 1.0 / distances
    weights /= weights.sum(axis=1, keepdims=True)
    return np.sum(weights * field[idx], axis=1)

points = np.array(mesh.points)
omega = smooth_all_field(points, omega, k_neighbors=8)

mesh["omega"] = omega
mesh.save("ref_les\\les_mesh_with_k.vtp")