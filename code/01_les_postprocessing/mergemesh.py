import pyvista as pv

path1 = "ref_les\\yNormal.vtp"
path2 = "ref_les\\zNormal.vtp"
mesh1 = pv.read(path1)
mesh2 = pv.read(path2)
merged_mesh = mesh1.merge(mesh2)
merged_mesh.save("ref_les\\les_mesh.vtp")
print("Merged mesh saved as 'les_mesh.vtp'")