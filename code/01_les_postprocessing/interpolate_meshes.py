import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree

les = pv.read("les_reference/les_mesh_with_k.vtp")
rans = pv.read("clustering_simp_corr_provafinale/rans_mesh.vtp")

tree = cKDTree(les.points)
dist, idx = tree.query(rans.points, k=1)
tol = 1e-8  # adjust as needed
if dist.max() > tol:
    raise ValueError(f"Meshes don't coincide within tol (max err {dist.max()})")

rans["k_les"] = les["k"][idx]
rans["omega_les"] = les["omega"][idx]
rans["k_sgs_les"] = les["k_sgs"][idx]
rans["k_res_les"] = les["k_res"][idx]
rans["nutEq"] = les["nutEq"][idx]
rans.save("dati_per_rete/rans_mesh_with_les_data_provafinale.vtp")
