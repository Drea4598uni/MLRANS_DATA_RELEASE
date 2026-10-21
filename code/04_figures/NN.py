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

import xz_maps

# Fig. 11: nu_t,eq, nu_t,mod (SST nu_t on the frozen LES mean fields, the
# training baseline) and nu_t,ML on the xz plane of the 30.8M training grid,
# stacked at full text width with one colour scale (same quantity)
mesh = pv.read(cc.PRED_CLU)
nut_maps = [(field, xz_maps.resample(mesh, field, (-5, 15), (0, 2.5))) for field in ('nutEq', 'nut', 'nut_pred')]
nut_top = max(float(np.nanmax(values)) for _, values in nut_maps)
xz_maps.map_stack(
    [{'values': values, 'tag': tag, 'title': title}
     for (_, values), tag, title in zip(nut_maps, 'abc', (r'$\nu_{t,eq}$', r'$\nu_{t,mod}$', r'$\nu_{t,ML}$'))],
    cc.OUT / 'nut_fields_xz.pdf',
    shared_colorbar={'cmap': 'viridis', 'clim': (0.0, nut_top), 'label': r'$\nu_t$ [m$^2$/s]'},
)


# Load the dataset
dataset = "data/05_neural_network/ml_rans_clustering/results/rans_with_predictions.vtu"
data = pv.read(dataset)

# Fields saved by the training script:
# - nut: reference viscosity used as the baseline in the reconstruction
# - nutEq: LES-equivalent viscosity
# - delta_pred: NN correction already inverse-transformed by y_scaler_nut.pkl
nut_base = data["nut"]
nut_eq = data["nutEq"]
delta_nut_pred = data["delta_pred"]
delta_nut_true = nut_eq - nut_base
nut_pred = data["nut_pred"]

# Reconstruct the prediction from the saved delta, as done during training.
nut_pred_from_delta = nut_base + delta_nut_pred

# Create a scatter plot of the true vs predicted values of nutEq
plt.figure(figsize=(6, 4))
plt.scatter(nut_eq, nut_pred, alpha=0.5, label="Predicted vs True", s=1, rasterized=True)
lims = [min(nut_eq.min(), nut_pred.min()), max(nut_eq.max(), nut_pred.max())]
plt.plot(lims, lims, 'r--', label="Ideal Fit")
plt.xlabel(r"True $\nu_{t,eq}$")
plt.ylabel(r"Predicted $\nu_{t,ML}$")
rect = mpatches.Rectangle((-1, -1), 42, 5, facecolor='none', edgecolor='blue', linestyle='--', label="Bias Region")
plt.gca().add_patch(rect)
plt.legend()
cc.save_figure("figures_output/NN_scatter_plot.png")
plt.close()

# Create a scatter plot of the true vs predicted delta
plt.figure(figsize=(6, 4))
plt.scatter(delta_nut_true, delta_nut_pred, alpha=0.5, label="Predicted vs True", s=1, rasterized=True)
lims = [
    min(delta_nut_true.min(), delta_nut_pred.min()),
    max(delta_nut_true.max(), delta_nut_pred.max()),
]
plt.plot(lims, lims, 'r--', label="Ideal Fit")
plt.xlabel(r"True $\Delta \nu_{t}$")
plt.ylabel(r"Predicted $\Delta \nu_{t,ML}$")
rect = mpatches.Rectangle((-35, -35), 43, 6, facecolor='none', edgecolor='blue', linestyle='--', label="Bias Region")
plt.gca().add_patch(rect)
plt.legend()
cc.save_figure("figures_output/NN_delta_scatter_plot.png")
plt.close()

# plot train val loss curves
path = "data/05_neural_network/ml_rans_clustering/results/training_history_nut.csv"
datas = pd.read_csv(path)
datas.plot(y=["loss", "val_loss"], logy=True, use_index=True)
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.title("Training and Validation Loss Curves")
plt.legend(["Training Loss", "Validation Loss"])
cc.save_figure("figures_output/NN_loss_curves.png")
plt.close()

datas.plot(y=["mae", "val_mae"], logy=True, use_index=True)
plt.xlabel("Epoch")
plt.ylabel("MAE")
# plt.title("Training and Validation MAE Curves")
plt.legend(["Training MAE", "Validation MAE"])
cc.save_figure("figures_output/NN_MAE_curves.png")
plt.close()

datas.plot(y=["rmse", "val_rmse"], logy=True, use_index=True)
plt.xlabel("Epoch")
plt.ylabel("RMSE")
# plt.title("Training and Validation RMSE Curves")
plt.legend(["Training RMSE", "Validation RMSE"])
cc.save_figure("figures_output/NN_RMSE_curves.png")
plt.close()
