import pyvista as pv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scienceplots
from PIL import Image
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc

plt.style.use(['science', 'ieee'])
# --- SEEP2026 template: force Times New Roman to match the manuscript body ---
plt.rcParams.update({
    'text.usetex': False,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
})
plt.rcParams['figure.dpi'] = 300
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['lines.linewidth'] = 1.2
plt.rcParams['lines.markersize'] = 4
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.frameon'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False

rotor_diameter = 126.0
OUTPUT_DIR = cc.OUT
# file-name suffix per inflow case (the 11.4 m/s maps keep their original names)
CASE_SUFFIX = {'11ms': '', '8ms': '_8ms', '15ms': '_15ms'}

def crop_to_low_z(mesh, z_max=2.5):
    return mesh.clip(normal='z', origin=(0.0, 0.0, z_max), invert=True)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
rotor_center = np.array([0.0, 0.0, 90.0]) / rotor_diameter
rotor_radius = 63.0 / rotor_diameter
rotor_line_radius = 2.5 / rotor_diameter
rotor_line_xz = pv.Line(
    rotor_center + np.array([0.0, 0.0, -rotor_radius]),
    rotor_center + np.array([0.0, 0.0, rotor_radius]),
).tube(radius=rotor_line_radius)
rotor_line_xy = pv.Line(
    rotor_center + np.array([0.0, -rotor_radius, 0.0]),
    rotor_center + np.array([0.0, rotor_radius, 0.0]),
).tube(radius=rotor_line_radius)
rotor_hub = pv.Sphere(radius=4.0 / rotor_diameter, center=rotor_center)

bounds_args_xz = {
    'xtitle': '',
    'ytitle': '',
    'ztitle': '',
    'fmt': '%.1fD',
    'font_size': 12 * cc.SCREENSHOT_SCALE,  # cube-axes labels are sized in pixels
    'font_family': 'times',
    'location': 'outer',
    'grid': None,
    'bounds': (-5, 15, 0, 0, 0, 2.5),
    'n_xlabels': 5,
    'n_zlabels': 3,
}

scalar_bar_args = {
    'title_font_size': 22,
    'label_font_size': 18,
    'font_family': 'times',
    'position_x': 0.2,
    'position_y': 0.205,
    'fmt': '%.0f',
}

def add_xz_axis_titles(plotter):
    plotter.add_text('x/D', position=cc.px((500, 270)), font_size=11, font='times', color='black')
    plotter.add_text('z/D', position=cc.px((75, 365)), font_size=11, font='times', color='black', orientation=90)

def add_scalar_bar_math_title(path, title):
    image = Image.open(path).convert('RGB')
    pixels = np.asarray(image)
    width, height = image.size
    dpi = 100 * cc.SCREENSHOT_SCALE

    y0 = int(0.72 * height)
    panel = pixels[y0:]
    color_mask = (
        (panel.max(axis=2) - panel.min(axis=2) > 40)
        & (panel.max(axis=2) < 245)
        & (panel.min(axis=2) < 230)
    )
    ys, xs = np.where(color_mask)
    if xs.size == 0:
        x_pos, y_pos = 0.75, 0.31
    else:
        left = int(xs.min())
        right = int(xs.max())
        top = y0 + int(ys.min())
        x_pos = ((left + right) / 2) / width
        y_pos = 1 - (top - 50 * cc.SCREENSHOT_SCALE) / height

    with plt.rc_context({'text.usetex': False, 'font.family': 'serif', 'font.serif': ['Times New Roman', 'Nimbus Roman No9 L', 'DejaVu Serif'], 'mathtext.fontset': 'stix'}):
        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.imshow(image)
        ax.axis('off')
        ax.text(
            x_pos,
            y_pos,
            title,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=16,
            color='black',
        )
        fig.savefig(path, dpi=dpi, pad_inches=0)
        plt.close(fig)

import xz_maps

# panels (b)-(d) of Figs. 12, 14 and 15: |U|, k and nu_t,ML of ML-RANS on the
# 15.9M grid, stacked at full text width (one file per inflow case)
# (k: colour scale set above z/D = 0.2, as in Fig. 16; the near-ground peak saturates)
PAPER_FIELDS = [
    (xz_maps.velocity_magnitude('U'), r'$|\vec{U}|$ [m/s]', 'b'),
    ('k', r'$k$ [m$^2$/s$^2$]', 'c'),
    ('nut', r'$\nu_{t,ML}$ [m$^2$/s]', 'd'),
]
for case in cc.CASES:
    mesh = pv.read(cc.solver_meshes(case, 'ML-RANS 15M')[0])
    panels = []
    for scalars, label, tag in PAPER_FIELDS:
        values = xz_maps.resample(mesh, scalars, (-5, 15), (0, 2.5))
        top = xz_maps.max_above(values, (0, 2.5)) if scalars == 'k' else float(np.nanmax(values))
        panels.append({'values': values, 'cmap': 'viridis', 'clim': (0.0, top), 'cbar_label': label, 'tag': tag,
                       'extend': 'max' if top < np.nanmax(values) else 'neither'})
    xz_maps.map_stack(panels, OUTPUT_DIR / f'predicted_fields_xz{CASE_SUFFIX[case]}.pdf')

fields = [
    ('U', r'$|U|$', 'predicted_U_field_xz'),
    ('nut', r'$\nu_{t,ML}$', 'predicted_nut_field_xz'),
    ('k', r'$k$', 'predicted_k_field_xz'),
]

# single pyvista maps of the same fields (not used in the paper any more)
for case in cc.CASES:
    mesh = pv.read(cc.solver_meshes(case, 'ML-RANS 15M')[0])
    cc.scale_to_diameters(mesh)
    mesh_low_z = crop_to_low_z(mesh)

    for scalars, title, name in fields:
        output_path = str(OUTPUT_DIR / f'{name}{CASE_SUFFIX[case]}.png')
        field_scalar_bar_args = scalar_bar_args.copy()
        field_scalar_bar_args['title'] = ' '
        clim = (0.0, float(np.nanmax(mesh_low_z[scalars])))
        # integer labels only when the range allows them (e.g. k at 8 m/s does not)
        if clim[1] < 10:
            field_scalar_bar_args['fmt'] = '%.1f'

        plotter = cc.hires_plotter()
        plotter.add_mesh(
            mesh_low_z,
            scalars=scalars,
            cmap='turbo',
            clim=clim,
            show_scalar_bar=True,
            scalar_bar_args=field_scalar_bar_args,
        )
        plotter.add_mesh(rotor_line_xz, color='black')
        plotter.add_mesh(rotor_hub, color='black')
        add_xz_axis_titles(plotter)
        cc.show_xz_bounds(plotter, bounds_args_xz)
        plotter.view_xz()
        plotter.enable_parallel_projection()
        plotter.screenshot(output_path)
        plotter.close()
        add_scalar_bar_math_title(output_path, title)
