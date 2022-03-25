from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import numpy as np
import os
import pandas as pd


# Parse user input
parser = Parser("test_cases/array/plot_inflow.py")
parser.parse_approach(default="fixed_mesh")
parsed_args = parser.parse_args()
approach = parsed_args.approach.split("_dwr")[0]

# Timesteps to take slices at
timesteps = [0, 20, 40, 80]
timesteps_per_cycle = 400

# Load data
configs = ("aligned", "staggered")
xvelocity = {}
for config in configs:
    xvelocity[config] = []
    run = "level5" if approach == "fixed_mesh" else "target10000"
    data_dir = f"outputs/{config}/{approach}/{run}/Velocity2d/extracts"
    for i in timesteps:
        idx = str(i)
        padded = "0"*(6 - len(idx)) + idx
        f = pd.read_csv(f"{data_dir}/inflow_{padded}.csv", usecols=["x-velocity"])
        xvelocity[config].append(np.array(f["x-velocity"]))

# Plot
x = np.arange(0, 1001)
fig, axes = plt.subplots()
for i, xva, xvs in zip(timesteps, xvelocity["aligned"], xvelocity["staggered"]):
    axes.plot(x[np.isfinite(xva)], xva[np.isfinite(xva)], color="C0", label="Aligned" if i == 0 else None)
    axes.plot(x[np.isfinite(xvs)], xvs[np.isfinite(xvs)], color="mediumseagreen", label="Staggered" if i == 0 else None)
    if approach != "fixed_mesh":
        note = r"$%.2f\,T_{\mathrm{tide}}$" % (1.0 + i / timesteps_per_cycle)
        axes.annotate(note, (1030, xva[-1]), fontsize=16, annotation_clip=False)
axes.set_xlim([0, 1000])
axes.set_xlabel(r"$y$-coordinate ($\mathrm{m}$)")
axes.set_ylabel(r"Inflow velocity ($\mathrm{m\,s}^{-1}$)")
axes.set_ylim([0, 3])
axes.grid(True)
axes.set_aspect(250)
lines, labels = axes.get_legend_handles_labels()
l = ["Fixed mesh" if approach == "fixed_mesh" else "Adaptive"]
axes.legend(lines[:1], l, loc="upper right", handlelength=0, handletextpad=0, fontsize=18)
plt.tight_layout()
plt.savefig(f"plots/inflow_{approach}.pdf")

# Plot legend separately
fname = "plots/legend_inflow.pdf"
if not os.path.exists(fname):
    fig2, axes2 = plt.subplots()
    legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=2)
    fig2.canvas.draw()
    axes2.set_axis_off()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    plt.savefig(fname, bbox_inches=bbox)
