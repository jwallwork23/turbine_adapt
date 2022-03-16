from thetis import create_directory
from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
from utils import time_integrate
from options import ArrayOptions
import h5py
import numpy as np
import os
import sys


# Parse arguments
parser = Parser(prog="test_cases/array/plot_power.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.add_argument(
    "mode",
    help="Should we use ramp data and/or subsequent run?",
    choices=["run", "ramp", "both"],
    default="run",
)
parser.add_argument(
    "--combine_plots",
    help="Plot the overall power as well as column-wise?",
    action="store_true",
)
parser.parse_setup()
parser.parse_approach(default="fixed_mesh")
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
config = parsed_args.configuration
combine_plots = parsed_args.combine_plots
approach = parsed_args.approach.split("_dwr")[0]
level = parsed_args.level
target = parsed_args.target_complexity
mode = parsed_args.mode
modes = ["ramp", "run"] if mode == "both" else [mode]
end_time = parsed_args.end_time
num_tidal_cycles = parsed_args.num_tidal_cycles
options = ArrayOptions(level=level, configuration=config, meshgen=True)
if end_time is None:
    end_time = options.ramp_time
    if mode != "ramp":
        end_time += num_tidal_cycles * options.tide_time

if approach == "fixed_mesh":
    run = f"level{level}"
else:
    run = f"target{target:.0f}"
output_dir = f"outputs/{config}/{approach}/{run}"
plot_dir = create_directory(f"plots/{config}/{approach}/{run}")

# Setup directories
power = np.array([]).reshape((0, 15))
time = np.array([]).reshape((0, 1))
for m in modes:
    ramp = m == "ramp"
    input_dir = output_dir + "/ramp" if ramp else output_dir
    fname = f"{input_dir}/diagnostic_turbine.hdf5"
    if not os.path.exists(fname):
        print(f"File {fname} does not exist")
        sys.exit(0)

    # Load data
    with h5py.File(fname, "r") as f:
        t = np.array(f["time"])
        if not ramp:
            t += options.tide_time
        time = np.vstack((time, t))
        power = np.vstack((power, np.array(f["current_power"])))
power = power[time.flatten() <= end_time, :] * 1030.0 / 1.0e06  # MW
time = time[time <= end_time]
p = power.copy()
t = time.flatten().copy()

# Calculate energy per halfcycle
dt = 0.5 * options.tide_time
energy = []
energy_time = []
while len(t) > 0:
    tnext = t[0] + dt
    energy.append(sum(time_integrate(p[t <= tnext, :], t[t <= tnext])))
    energy_time.append(t[0] + 0.2 * dt)
    p = p[t >= tnext]
    t = t[t >= tnext]

# Convert to cycle time
end_time /= options.tide_time
time /= options.tide_time
energy_time = np.array(energy_time) / options.tide_time
energy = np.array(energy) / 3600.0  # MWh

colours = ["b", "C0", "mediumturquoise", "mediumseagreen", "g"]
kw = {"linewidth": 1.0}
ticks = []
eps = 1.0e-05
if mode != "run":
    ticks += [0, 0.25, 0.5, 0.75, 1]
else:
    ticks += list(np.arange(1, end_time + eps, 0.125))
# NOTE: Assumes ramp is just one tidal cycle
if mode == "both":
    ticks += list(np.arange(1, end_time + eps, 0.25))

# Plot power output of each column
figsize = (4.4 + 2 * (end_time - 1 if mode == "run" else end_time), 4.8)
fig, axes = plt.subplots(figsize=figsize)
for i in range(5):
    _power = power[:, 3 * i] + power[:, 3 * i + 1] + power[:, 3 * i + 2]
    axes.plot(time, _power, label=f"{i+1}", color=colours[i], **kw)
if combine_plots:
    axes.plot(
        time,
        np.sum(power, axis=1) * 1030.0 / 1.0e06,
        "--",
        label="Overall",
        color="gray",
        **kw,
    )
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.set_xticks(ticks)
axes.set_xlim([ticks[0], ticks[-1]])
ymax = 40 if combine_plots else 15 if config == "aligned" else 18
if mode == "both":
    axes.vlines(1, 0, ymax, "k")
    axes.vlines(1.5, 0, ymax, "k")
axes.set_ylim([0, ymax])
axes.grid(True)
plt.tight_layout()
cmb = "_combined" if combine_plots else ""
plt.savefig(f"{plot_dir}/{config}_power_output_column_{run}_{mode}{cmb}.pdf")

# Plot legend separately
fname = "plots/legend_column.pdf"
if not os.path.exists(fname):
    fig2, axes2 = plt.subplots()
    lines, labels = axes.get_legend_handles_labels()
    legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=5)
    fig2.canvas.draw()
    axes2.set_axis_off()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    plt.savefig(fname, bbox_inches=bbox)

# Plot total power output
fig, axes = plt.subplots(figsize=figsize)
ymax = 40 if config == "aligned" else 60
if mode == "both":
    axes.vlines(1, 0, ymax, "k")
    axes.vlines(1.5, 0, ymax, "k")
axes.plot(time, np.sum(power, axis=1), **kw)
h = 5 if config == "aligned" else 7.5
for t, e in zip(energy_time, energy):
    axes.annotate(f"{e:.2f} MWh", (t, h), fontsize=10)
axes.set_xticks(ticks)
axes.set_yticks(np.arange(10, ymax + eps, 10))
axes.set_xlim([ticks[0], ticks[-1]])
axes.set_ylim([0, ymax])
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/{config}_total_power_output_{run}_{mode}.pdf")
