"""Fig. 3: eddy-viscosity fields of the three CFD simulations on the xz plane
through the rotor axis -- (a) standard RANS nu_t, (b) LES-equivalent nu_t,eq,
(c) modified RANS nu_t,mod (SST nu_t on the frozen LES mean fields) -- with
white isolines of the velocity magnitude (RANS velocity in (a), LES mean
velocity in (b) and (c)). One colour scale for the three maps.

Run from the package root:
    py methodology_fields_fig.py
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc  # noqa: E402  (sets the working directory)
import xz_maps  # noqa: E402

PATH_RANS_ORIGINAL_XZ = Path('data') / '02_rans_reference' / 'yNormal_11ms.vtp'
PATH_LES_AND_MODIFIED = Path('data') / '03_training_data' / 'rans_mesh_with_les_data_.vtu'
OUTPUT = cc.OUT / 'cfds_nut_xz.pdf'
X_RANGE, Z_RANGE = (-5, 15), (0, 2.5)

rans = pv.read(PATH_RANS_ORIGINAL_XZ)
les_and_modified = pv.read(PATH_LES_AND_MODIFIED)
speed = xz_maps.velocity_magnitude('U')

panels = []
for mesh, field, title, tag in (
    (rans, 'nut', r'$\nu_t$ (RANS)', 'a'),
    (les_and_modified, 'nutEq', r'$\nu_{t,eq}$', 'b'),
    (les_and_modified, 'nut', r'$\nu_{t,mod}$', 'c'),
):
    panels.append({
        'values': xz_maps.resample(mesh, field, X_RANGE, Z_RANGE),
        'speed': xz_maps.resample(mesh, speed, X_RANGE, Z_RANGE),
        'title': title,
        'tag': tag,
    })

# same isolines in the three maps, spanning the bulk of the velocity range
speeds = np.concatenate([panel['speed'].ravel() for panel in panels])
levels = np.linspace(np.nanpercentile(speeds, 15), np.nanpercentile(speeds, 95), 7)
for panel in panels:
    panel['contours'] = (panel.pop('speed'), levels)

top = max(float(np.nanmax(panel['values'])) for panel in panels)
xz_maps.map_stack(
    panels, OUTPUT, X_RANGE, Z_RANGE,
    shared_colorbar={'cmap': 'viridis', 'clim': (0.0, top), 'label': r'$\nu_t$ [m$^2$/s]'},
)
