import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import scienceplots
sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc
plt.style.use(['science', 'ieee'])
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

### elbow plot
elbow_df = pd.read_csv(cc.CLUSTER_ELBOW)
NC = elbow_df['NC']
inertia = elbow_df['Inertia']

plt.figure()
plt.plot(NC, inertia, marker='o', linestyle='--')
plt.scatter([7], inertia[NC == 7], marker='x', color='red', label=r'Selected $K$', zorder=5, s=75)
plt.xlabel(r'Number of clusters $K$')
plt.ylabel('Inertia [-]')
plt.legend()
plt.grid(axis='x')
cc.save_figure(cc.OUT / 'elbow_plot.png')
plt.close()

### PCA plot
pca_df = pd.read_csv(cc.CLUSTER_PCA_VARIANCE)
PCs = pca_df.iloc[:, 0]
explained_variance_ratio = pca_df['Explained Variance Ratio']
x_pcs = range(1, len(PCs) + 1)
cumulative_explained_variance_ratio = []
for i in range(len(PCs)):
    cumulative_explained_variance_ratio.append(explained_variance_ratio[:i+1].sum())
    
plt.figure()
plt.plot(x_pcs, cumulative_explained_variance_ratio, marker='o', label='Cumulative Explained Variance Ratio')
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance ratio [-]')
plt.xticks(ticks=x_pcs)
plt.grid(axis='y')
cc.save_figure(cc.OUT / 'pca_explained_variance_ratio.png')
plt.close() 

### plot features weights on first 6 PCs
dict = {
    'Q': r'$Q$',
    'Rrr': r'$R_{rr}$',
    'Rrt': r'$R_{rt}$',
    'Rtt': r'$R_{tt}$',
    'Rxr': r'$R_{xr}$',
    'Rxt': r'$R_{xt}$',
    'Rxx': r'$R_{xx}$',
    'Ur': r'$U_\rho$',
    'Ut': r'$U_\theta$',
    'Ux': r'$U_x$',
    'k': r'$k$',
    'nut': r'$\nu_t$',
    'p': r'$p$',
    'vort_r': r'$\omega_r$',
    'vort_t': r'$\omega_t$',
    'vort_ax': r'$\omega_{ax}$',
    'ax': r'$x$',
    'rho': r'$\rho$',
    'theta': r'$\theta$'
}

features_weights_df = pd.read_csv(cc.CLUSTER_PCA_IMPORTANCE)
# U_cyl = (U_x, U_theta, U_rho) as in the text, then the other features
order = ['Q', 'Ux', 'Ut', 'Ur', 'k', 'nut', 'p', 'ax', 'rho', 'theta']
features = [f for f in order if f in features_weights_df.columns] + [f for f in features_weights_df.columns if f not in order]
weights = features_weights_df[features].values
plt.figure()
for i in range(len(weights)):
    plt.bar(features, weights[i], color=plt.cm.tab10(i), alpha=0.5)
plt.ylabel('Weight on the first 6 PCs [-]')
plt.xticks(ticks=range(len(features)), labels=[dict[feat] for feat in features], rotation=45, ha='right')
plt.tight_layout()
cc.save_figure(cc.OUT / 'pca_features_weights.png')
plt.close()

### plot clustering results (Fig. 9): hub-height xy plane and xz plane through
### the rotor axis, cropped to the wake region (outside it every cell is in cluster 1)
import pyvista as pv
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
import xz_maps

CLUSTER_COLORS = ['#1f77b4', '#ff7f0e', "#bc1f1f", "#49ff01", "#8c704b", "#0be4d2", '#bcbd22']
cluster_mesh = pv.read(cc.CLUSTER_MESH)
views = [
    ('xy', (-2.5, 2.5), 'a'),
    ('xz', (0.0, 2.5), 'b'),
]
cmap = ListedColormap(CLUSTER_COLORS)
norm = BoundaryNorm(np.arange(0.5, 8.5), cmap.N)
x_range = (-5, 15)
left, right, top, gap, bottom, legend_h = 0.45, 0.1, 0.08, 0.32, 0.64, 0.25
map_w = xz_maps.TEXTWIDTH_IN - left - right
heights = [map_w * (v[1] - v[0]) / (x_range[1] - x_range[0]) for _, v, _ in views]
fig_h = top + sum(heights) + (gap + 0.3) * (len(views) - 1) + bottom + legend_h
with plt.rc_context(xz_maps.STYLE):
    fig = plt.figure(figsize=(xz_maps.TEXTWIDTH_IN, fig_h))
    y_top = fig_h - top
    for i, ((plane, v_range, tag), h) in enumerate(zip(views, heights)):
        ax = fig.add_axes((left / xz_maps.TEXTWIDTH_IN, (y_top - h) / fig_h, map_w / xz_maps.TEXTWIDTH_IN, h / fig_h))
        ids = xz_maps.resample(cluster_mesh, 'ID', x_range, v_range, plane=plane, categorical=True)
        ax.imshow(ids, origin='lower', extent=(*x_range, *v_range), cmap=cmap, norm=norm,
                  interpolation='nearest', aspect='equal', rasterized=True)
        xz_maps._draw_rotor(ax, plane)
        last = i == len(views) - 1
        xz_maps._format_axes(ax, x_range, v_range, plane, last=True)
        xz_maps._tag(fig, ax, tag, True, fig_h)
        y_top -= h + gap + (0.3 if not last else 0)
    handles = [Patch(facecolor=c, edgecolor='none', label=f'Cluster {i}') for i, c in enumerate(CLUSTER_COLORS, start=1)]
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               handlelength=1.2, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))
    cc.save_figure(cc.OUT / 'clustering_result.png', fig, bbox_inches=None, dpi=600)
    plt.close(fig)
