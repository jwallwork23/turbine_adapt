from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import numpy as np
import os
import pandas as pd


# Parse user input
parser = Parser("test_cases/array/plot_inflow.py")
parser.parse_approach()
parsed_args = parser.parse_args()
approach = parsed_args.approach

# Timesteps to take slices at
timesteps = [10, 30, 50, 100]
timesteps_per_cycle = 400

# Model parameters
density = 1030.0
W = 5
D = 20
H = 50
turbine_footprint = W * D
turbine_swept = np.pi * (D / 2) ** 2
turbine_cross_section = D * H
Ct = 2.985
coeff = 0.5 * (1.0 + np.sqrt(1.0 - Ct * turbine_swept / turbine_cross_section))
correction = 1 / coeff ** 2
Ct *= correction
ct = 0.5 * Ct * turbine_swept / turbine_footprint

# Load data
configs = ("aligned", "staggered")
xvelocity = {}
for config in configs:
    xvelocity[config] = []
    run = "level5" if approach == "fixed_mesh" else "target10000"
    data_dir = f"outputs/{config}/{approach}/{run}/Velocity2d/extracts"
    for i in timesteps:
        f = pd.read_csv(f"{data_dir}/inflow_{i:06d}.csv", usecols=["x-velocity"])
        xvelocity[config].append(np.array(f["x-velocity"]))

# Plot inflow velocity
x = np.arange(0, 1001)
fig, axes = plt.subplots()
for i, xva, xvs in zip(timesteps, xvelocity["aligned"], xvelocity["staggered"]):
    axes.plot(x[np.isfinite(xva)], xva[np.isfinite(xva)], color="C0", label="Aligned" if i == 0 else None)
    axes.plot(x[np.isfinite(xvs)], xvs[np.isfinite(xvs)], color="mediumseagreen", label="Staggered" if i == 0 else None)
    note = r"$%.3f\,T_{\mathrm{tide}}$" % (1.0 + i / timesteps_per_cycle)
    axes.annotate(note, (1030, 0.5 * (xva[-1] + xvs[-1])), fontsize=16, annotation_clip=False)
axes.set_xlim([0, 1000])
axes.set_xlabel(r"$y$-coordinate ($\mathrm{m}$)")
axes.set_ylabel(r"Normal velocity ($\mathrm{m\,s}^{-1}$)")
axes.grid(True)
lines, labels = axes.get_legend_handles_labels()
plt.tight_layout()
plt.savefig(f"plots/inflow_velocity_{approach}.pdf")

# Plot legend separately
fname = "plots/legend_inflow.pdf"
if not os.path.exists(fname):
    fig2, axes2 = plt.subplots()
    legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=2)
    fig2.canvas.draw()
    axes2.set_axis_off()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    plt.savefig(fname, bbox_inches=bbox)

# Plot inflow dynamic pressure
x = np.arange(0, 1001)
fig, axes = plt.subplots()
for i, xva, xvs in zip(timesteps, xvelocity["aligned"], xvelocity["staggered"]):
    dpa = 0.5 * density * xva[np.isfinite(xva)] ** 2 / 1000
    axes.plot(x[np.isfinite(xva)], dpa, color="C0", label="Aligned" if i == 0 else None)
    dps = 0.5 * density * xvs[np.isfinite(xvs)] ** 2 / 1000
    axes.plot(x[np.isfinite(xvs)], dps, color="mediumseagreen", label="Staggered" if i == 0 else None)
    note = r"$%.3f\,T_{\mathrm{tide}}$" % (1.0 + i / timesteps_per_cycle)
    axes.annotate(note, (1030, 0.5 * (dpa[-1] + dps[-1])), fontsize=16, annotation_clip=False)
axes.set_xlim([0, 1000])
axes.set_xlabel(r"$y$-coordinate ($\mathrm{m}$)")
axes.set_ylabel(r"Dynamic pressure ($\mathrm{kPa}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"plots/inflow_dynamic_pressure_{approach}.pdf")

# Plot theoretical inflow power
x = np.arange(0, 1001)
fig, axes = plt.subplots()
for i, xva, xvs in zip(timesteps, xvelocity["aligned"], xvelocity["staggered"]):
    print(density, ct, turbine_footprint, np.abs(xva[np.isfinite(xva)]).max() ** 3)
    pa = density * ct * turbine_footprint * np.abs(xva[np.isfinite(xva)]) ** 3 / 1.0e06
    axes.plot(x[np.isfinite(xva)], pa, color="C0", label="Aligned" if i == 0 else None)
    print(density, ct, turbine_footprint, np.abs(xva[np.isfinite(xvs)]).max())
    ps = density * ct * turbine_footprint * np.abs(xvs[np.isfinite(xvs)]) ** 3 / 1.0e06
    axes.plot(x[np.isfinite(xvs)], ps, color="mediumseagreen", label="Staggered" if i == 0 else None)
    note = r"$%.3f\,T_{\mathrm{tide}}$" % (1.0 + i / timesteps_per_cycle)
    axes.annotate(note, (1030, 0.5 * (pa[-1] + ps[-1])), fontsize=16, annotation_clip=False)
axes.set_xlim([0, 1000])
axes.set_xlabel(r"$y$-coordinate ($\mathrm{m}$)")
axes.set_ylabel(r"Theoretical power ($\mathrm{MW}$)")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"plots/inflow_power_{approach}.pdf")
