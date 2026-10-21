"""Regenerate every data-driven figure and table of the paper from the files in data/.

    python code/04_figures/run_all.py

The individual scripts write to figures_output/; at the end the files used in
the manuscript are also copied to figures_output/paper/ under their figure
number (fig9.pdf is Fig. 9; for Figs. 12, 14, 15 and 18, which the article
assembles from two parts, fig12a.pdf and fig12b.pdf are the two parts) and the
region-averaged deficit errors under table4_zonal_deficit_error.csv.
Figures 1, 2, 4, 5 and 6 are schematics or mesh renderings and are not
generated from data.
"""
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import case_config as cc  # noqa: E402  (sets the working directory to the package root)

SCRIPTS = [
    "methodology_fields_fig.py",
    "clustering_fig.py",
    "NN.py",
    "nn_training_figure.py",
    "plot_fields.py",
    "plot_error_field.py",
    "plot_velocity_deficit.py",
    "zonal_deficit_error.py",
    "plot_k_profiles.py",
    "plot_cross_planes.py",
    "forces_cp_ct.py",
]

O = "figures_output"
ML = f"{O}/ml_results"

# paper file  <-  file written by the scripts
PAPER_FIGURES = {
    "fig3.pdf": f"{O}/cfds_nut_xz.pdf",
    "fig7a.pdf": f"{O}/pca_explained_variance_ratio.pdf",
    "fig7b.pdf": f"{O}/pca_features_weights.pdf",
    "fig8.pdf": f"{O}/elbow_plot.pdf",
    "fig9.pdf": f"{O}/clustering_result.pdf",
    "fig10.pdf": f"{O}/nn_training.pdf",
    "fig11.pdf": f"{O}/nut_fields_xz.pdf",
    "fig12a.pdf": f"{ML}/11ms/deficit/ml_models_comparison/velocity/comparison.pdf",
    "fig12b.pdf": f"{O}/predicted_fields_xz.pdf",
    "fig13.pdf": f"{ML}/11ms/error_fields/error_U_paper.pdf",
    "fig14a.pdf": f"{ML}/8ms/deficit/ml_models_comparison/velocity/comparison.pdf",
    "fig14b.pdf": f"{O}/predicted_fields_xz_8ms.pdf",
    "fig15a.pdf": f"{ML}/15ms/deficit/ml_models_comparison/velocity/comparison.pdf",
    "fig15b.pdf": f"{O}/predicted_fields_xz_15ms.pdf",
    "fig16.pdf": f"{ML}/k_profiles_5m.pdf",
    "fig17.pdf": f"{ML}/cross_plane_metrics.pdf",
    "fig18a.pdf": f"{O}/forces_cp_ct.pdf",
    "fig18b.pdf": f"{O}/forces_profile.pdf",
    "table4_zonal_deficit_error.csv": f"{ML}/zonal_deficit_error.csv",
}


def main():
    # one process per script: several scripts set the matplotlib style globally,
    # which would otherwise carry over to the following ones
    for script in SCRIPTS:
        print(f"=== {script}", flush=True)
        subprocess.run([sys.executable, str(HERE / script)], cwd=cc.ROOT, check=True)

    paper = Path(O) / "paper"
    paper.mkdir(parents=True, exist_ok=True)
    missing = []
    for target, source in PAPER_FIGURES.items():
        if Path(source).exists():
            shutil.copy2(source, paper / target)
        else:
            missing.append(source)
    print(f"Copied {len(PAPER_FIGURES) - len(missing)} paper files to {paper}")
    if missing:
        sys.exit("Missing outputs:\n  " + "\n  ".join(missing))


if __name__ == "__main__":
    main()
