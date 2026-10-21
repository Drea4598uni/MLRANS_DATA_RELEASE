"""Training of the MLP with and without clustering (one figure, review A.8/C):
(a) training and validation loss of the two models, (b) nu_t,ML against
nu_t,eq on the validation cells, model with clustering, (c) the same without
clustering.

The validation cells are rebuilt exactly as in the training script
(code/02_training/train_ml_rans_clustering.py):
cells with ax_d >= -2.5 and 0 <= z_d <= 4, then train_test_split(test_size=0.2,
random_state=42). The rebuilt split reproduces the metrics stored in
metrics_nut.json (R2, RMSE and MAE, training and validation, both models).

Run from the package root:
    python code/04_figures/nn_training_figure.py
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
import case_config as cc  # noqa: E402
import xz_maps  # noqa: E402  (shared print style)

OUTPUT = Path("figures_output/nn_training.pdf")
MODELS = {
    "with clustering": (cc.PRED_CLU, cc.PRED_CLU_HISTORY, "black"),
    "without clustering": (cc.PRED_NOCLU, cc.PRED_NOCLU_HISTORY, "magenta"),
}
TRAIN_WINDOW = dict(min_ax_d=-2.5, min_z_d=0.0, max_z_d=4.0)


def validation_cells(mesh):
    ax_d, z_d = np.asarray(mesh["ax_d"]), np.asarray(mesh["z_d"])
    kept = np.flatnonzero((ax_d >= TRAIN_WINDOW["min_ax_d"]) & (z_d >= TRAIN_WINDOW["min_z_d"])
                          & (z_d <= TRAIN_WINDOW["max_z_d"]))
    train, val = train_test_split(kept, test_size=0.2, random_state=42)
    return train, val


def metrics(true, pred):
    err = pred - true
    return (1 - np.sum(err**2) / np.sum((true - true.mean()) ** 2), np.sqrt(np.mean(err**2)), np.mean(np.abs(err)))


def main():
    style = dict(xz_maps.STYLE)
    style["savefig.bbox"] = None
    with plt.rc_context(style):
        width = xz_maps.TEXTWIDTH_IN
        fig = plt.figure(figsize=(width, 2.45))
        axes = [fig.add_axes((x, 0.27, w, 0.66)) for x, w in ((0.075, 0.36), (0.53, 0.2), (0.785, 0.2))]
        ax_loss, scatter_axes = axes[0], axes[1:]
        top = 0.0
        rows = []
        for (label, (mesh_path, history_path, color)), ax in zip(MODELS.items(), scatter_axes):
            history = pd.read_csv(history_path)
            epochs = np.arange(1, len(history) + 1)
            ax_loss.plot(epochs, history["loss"], color=color, lw=0.9, label=f"Training, {label}")
            ax_loss.plot(epochs, history["val_loss"], color=color, lw=0.9, ls="--", label=f"Validation, {label}")
            mesh = pv.read(mesh_path)
            true, pred = np.asarray(mesh["nutEq"]), np.asarray(mesh["nut_pred"])
            train, val = validation_cells(mesh)
            r2, rmse, mae = metrics(true[val], pred[val])
            rows.append((label, *metrics(true[train], pred[train]), r2, rmse, mae))
            ax.scatter(true[val], pred[val], s=0.6, color=color, alpha=0.35, lw=0, rasterized=True)
            top = max(top, true[val].max(), pred[val].max())
            ax.text(0.04, 0.96, rf"$R^2={r2:.4f}$", transform=ax.transAxes, ha="left", va="top", fontsize=7)
            ax.set_xlabel(r"$\nu_{t,eq}$ [m$^2$/s]")
            ax.set_ylabel(r"$\nu_{t,ML}$ [m$^2$/s]", labelpad=1)
        lim = (min(-1.0, 0.0), np.ceil(top / 5) * 5)
        for ax in scatter_axes:
            ax.plot(lim, lim, color="red", lw=0.7, ls="--")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
        ax_loss.set_yscale("log")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss (normalised) [-]")
        ax_loss.legend(frameon=False, fontsize=6.5, loc="upper right")
        for ax, tag in zip(axes, "abc"):
            box = ax.get_position()
            fig.text(box.x0 + box.width / 2, 0.025, f"({tag})", ha="center", va="bottom", fontsize=xz_maps.TAG_SIZE)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT)
        fig.savefig(OUTPUT.with_suffix(".png"), dpi=300)
        plt.close(fig)
    print("model, train R2/RMSE/MAE, validation R2/RMSE/MAE")
    for row in rows:
        print(row[0], " ".join(f"{v:.4f}" for v in row[1:]))


if __name__ == "__main__":
    main()
