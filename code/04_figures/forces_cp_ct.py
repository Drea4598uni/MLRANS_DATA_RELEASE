import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots 
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

alpha_df = pd.read_csv('data/06_solver_results/11ms/CSV_files/timestep_sweep/11_4_m_s/timestep_sweep_11_4_m_s_alpha_distribution.csv')
Cp_df = pd.read_csv('data/06_solver_results/11ms/CSV_files/timestep_sweep/11_4_m_s/timestep_sweep_11_4_m_s_Cp_time_series.csv')
Ct_df = pd.read_csv('data/06_solver_results/11ms/CSV_files/timestep_sweep/11_4_m_s/timestep_sweep_11_4_m_s_Ct_time_series.csv')
Energy_df = pd.read_csv('data/06_solver_results/11ms/CSV_files/timestep_sweep/11_4_m_s/timestep_sweep_11_4_m_s_Energy_spectrum.csv')
force_df = pd.read_csv('data/06_solver_results/11ms/CSV_files/timestep_sweep/11_4_m_s/timestep_sweep_11_4_m_s_force_profile.csv')
Uref=11.4

# the two paper figures are drawn at their printed size (full text width),
# hence 7-8 pt fonts instead of the 6x4 in defaults above
PRINT_STYLE = {
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'lines.markersize': 3,
}

### plot Cp and Ct values against references

reference_cp = pd.read_csv('data/07_literature_reference/Cp_ref.csv')
reference_ct = pd.read_csv('data/07_literature_reference/c_t_new.csv') 
# Rotor coefficients of ML-RANS on the 4.2M grid at the rated condition (11.4 m/s),
# time-averaged over the last flow-through time (start-up transient discarded),
# from turbinesFoam's turbine1.csv (columns: cp = power, cd = thrust, ct = torque;
# data/06_solver_results/rotor_coefficients.csv).
# The values are divided by the air density, as in the authors' post-processing.
RHO_AIR = 1.225
rotor = pd.read_csv('data/06_solver_results/rotor_coefficients.csv')
ml = rotor[rotor['model'] == 'ML-RANS_4.2M'].set_index('case').loc['11ms']   # rated condition only
with plt.rc_context(PRINT_STYLE):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax1.plot(reference_cp['wind'], reference_cp['cp'], linestyle='--', label='Jonkman et al. (2009)')
    ax1.scatter(Uref, ml['cp'] / RHO_AIR, zorder=5)
    ax2.plot(reference_ct['tsr'], reference_ct['ct'], linestyle='--', label='Jonkman et al. (2009)')
    ax2.scatter(ml['tsr'], ml['cq'] / RHO_AIR, marker='s', zorder=5)
    ax1.set_xlabel(r'$U_\infty$ [m/s]')
    ax2.set_xlabel(r'$\lambda$ [-]')
    ax1.set_ylabel(r'$C_p$ [-]')
    ax2.set_ylabel(r'$C_q$ [-]')
    # one legend under the two panels (a legend marker inside a panel was read as data)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color='black', linestyle='--', label='Jonkman et al. (2009)'),
               Line2D([], [], color='black', marker='o', linestyle='none', label=r'ML-RANS 4.2M, $C_p$ at 11.4 m/s'),
               Line2D([], [], color='black', marker='s', linestyle='none', label=r'ML-RANS 4.2M, $C_q$ at $\lambda = 7$')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    cc.save_figure('figures_output/forces_cp_ct.png', fig)
    plt.close()

    ### plot force profile

    ref_norm_force = pd.read_csv('data/07_literature_reference/trold_norm.csv')
    ref_tan_force = pd.read_csv('data/07_literature_reference/trold_tang.csv')
    fig, ax = plt.subplots(1,2, figsize=(6.5, 2.7))
    ax[0].plot(force_df['normal']/(Uref**2)/(63)/1.225*(32/63), force_df['r'], label='ML-RANS', marker='o')
    ax[1].plot(force_df['tangential']/(Uref**2)/(63)/1.225*(32/63), force_df['r'], label='ML-RANS', marker='s')
    ax[0].plot(ref_norm_force['f'], ref_norm_force['r'], linestyle='--', label='Troldborg et al. (2015)')
    ax[1].plot(ref_tan_force['f'], ref_tan_force['r'], linestyle='--', label='Troldborg et al. (2015)')
    ax[0].set_ylabel('r/R [-]')
    ax[1].set_ylabel('r/R [-]')
    ax[0].set_xlabel(r'$F_n/(\rho U_\infty^2 R^2)$ [-]')
    ax[1].set_xlabel(r'$F_t/(\rho U_\infty^2 R^2)$ [-]')
    # one legend under the two panels: inside them it covered the profiles
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    cc.save_figure('figures_output/forces_profile.png', fig)
    plt.close()

### tangential-only force profile (short-paper version)
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(force_df['tangential']/(Uref**2)/(63)/1.225*(32/63), force_df['r'], label=r'$F_t$', marker='s')
ax.plot(ref_tan_force['f'], ref_tan_force['r'], label=r'Troldborg et al. (2015)', linestyle='--')
ax.set_ylabel('r/R')
ax.set_xlabel(r'$F_t$/$\rho U^2 R^2$')
ax.legend()
cc.save_figure('figures_output/forces_profile_tangential.png', fig)
plt.close()
