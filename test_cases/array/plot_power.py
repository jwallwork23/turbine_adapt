from thetis import create_directory
from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


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
parser.parse_setup()
parser.parse_approach()
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
config = parsed_args.configuration
approach = parsed_args.approach
level = parsed_args.level
target = parsed_args.target_complexity
mode = parsed_args.mode
modes = ["ramp", "run"] if mode == "both" else [mode]

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

    # Load data
    with h5py.File(os.path.join(input_dir, "diagnostic_turbine.hdf5"), "r") as f:
        t = np.array(f["time"])
        if not ramp:
            t += 4464.0
        t /= 4464.0
        time = np.vstack((time, t))
        power = np.vstack((power, np.array(f["current_power"])))

colours = ["b", "C0", "mediumturquoise", "mediumseagreen", "g"]
kw = {"linewidth": 1.0}

# Plot power output of each turbine
fig, axes = plt.subplots()
for i in range(15):
    _power = power[:, i] * 1030.0 / 1.0e06
    axes.plot(time, _power, label=f"Turbine {i+1}", color=colours[i // 3], **kw)
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_xlim([time[0], time[-1]])
if mode == "ramp":
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1])
elif mode == "run":
    axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
else:
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/power_output_{mode}.pdf")

# Plot power output of each column
fig, axes = plt.subplots()
for i in range(5):
    _power = power[:, 3 * i] + power[:, 3 * i + 1] + power[:, 3 * i + 2]
    axes.plot(time, _power * 1030.0 / 1.0e06, label=f"{i+1}", color=colours[i], **kw)
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.set_xlim([time[0], time[-1]])
if mode == "ramp":
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1])
elif mode == "run":
    axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
else:
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
axes.set_ylim([0, 15])
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/power_output_column_{mode}.pdf")

# Plot legend separately
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=5)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
plt.savefig(f"{plot_dir}/legend_column.pdf", bbox_inches=bbox)

# Plot total power output
fig, axes = plt.subplots()
axes.plot(time, np.sum(power, axis=1) * 1030.0 / 1.0e06, label=f"Turbine {i}", **kw)
axes.set_xlim([time[0], time[-1]])
if mode == "ramp":
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1])
elif mode == "run":
    axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
else:
    axes.set_xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5])
axes.set_xlabel(r"Time [$\mathrm s$]")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/total_power_output_{mode}.pdf")
