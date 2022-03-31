from thetis import create_directory
from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
from utils import time_integrate
from options import ArrayOptions
import h5py
import numpy as np
import os
import sys


eps = 1.0e-05


def get_data(config, modes, namespace):
    """
    :arg config: configuration, from 'aligned' and 'staggered'
    :arg modes: a list of simulation modes, from 'ramp' and 'run'
    :arg namespace: a :class:`NameSpace` of user input
    """
    level = namespace.level
    approach = namespace.approach
    end_time = namespace.end_time

    options = ArrayOptions(level=level, configuration=config, meshgen=True)
    if end_time is None:
        end_time = options.ramp_time
        if mode != "ramp":
            end_time += namespace.num_tidal_cycles * options.tide_time

    # Load data
    if approach == "fixed_mesh":
        run = f"level{level}"
    else:
        run = f"target{namespace.target_complexity:.0f}"
    output_dir = f"outputs/{config}/{approach}/{run}"
    power = np.array([]).reshape((0, 15))
    time = np.array([]).reshape((0, 1))
    for m in modes:
        ramp = m == "ramp"
        input_dir = output_dir + "/ramp" if ramp else output_dir
        fname = f"{input_dir}/diagnostic_turbine.hdf5"
        if not os.path.exists(fname):
            print(f"File {fname} does not exist")
            sys.exit(0)
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

    # Plot formatting
    ticks = []
    if mode != "run":
        ticks += [0, 0.25, 0.5, 0.75, 1]
    else:
        ticks += list(np.arange(1, end_time + eps, 0.125))
    # NOTE: Assumes ramp is just one tidal cycle
    if mode == "both":
        ticks += list(np.arange(1, end_time + eps, 0.25))
    figsize = (4.4 + 2 * (end_time - 1 if mode == "run" else end_time), 4.8)

    return power, time, energy, energy_time, end_time, run, ticks, figsize


# Parse arguments
parser = Parser(prog="test_cases/array/plot_power.py")
parser.add_argument(
    "configurations",
    nargs="+",
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
configs = parsed_args.configurations
if len(configs) == 0:
    print("Nothing to plot.")
    sys.exit(0)
approach = parsed_args.approach.split("_dwr")[0]
mode = parsed_args.mode
modes = ["ramp", "run"] if mode == "both" else [mode]
colours = ["b", "C0", "mediumturquoise", "mediumseagreen", "g"]
kw = {"linewidth": 1.0}

# Load data
power, time, energy, energy_time, end_time, run, ticks, figsize = get_data(configs[0], modes, parsed_args)

# Plot power output of each column
for config in configs:
    fig, axes = plt.subplots(figsize=figsize)
    for i in range(5):
        _power = power[:, 3 * i] + power[:, 3 * i + 1] + power[:, 3 * i + 2]
        axes.plot(time, _power, label=f"{i+1}", color=colours[i], **kw)
    if parsed_args.combine_plots:
        axes.plot(
            time,
            np.sum(power, axis=1),
            "--",
            label="Overall",
            color="gray",
            **kw,
        )
    axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
    axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
    axes.set_xticks(ticks)
    axes.set_yticks([5, 10, 15])
    axes.set_xlim([ticks[0], ticks[-1]])
    ymax = 40 if parsed_args.combine_plots else 15
    if mode == "both":
        axes.vlines(1, 0, ymax, "k")
        axes.vlines(1.5, 0, ymax, "k")
    axes.set_ylim([0, ymax])
    axes.grid(True)
    lines, labels = axes.get_legend_handles_labels()
    l = ["\n".join([config.capitalize(), "(fixed)" if approach == "fixed_mesh" else "(adaptive)"])]
    axes.legend(lines[:1], l, loc="upper right", handlelength=0, handletextpad=0, fontsize=18)
    plt.tight_layout()
    cmb = "_combined" if parsed_args.combine_plots else ""
    plot_dir = create_directory(f"plots/{config}/{approach}/{run}")
    plt.savefig(f"{plot_dir}/{config}_power_output_column_{run}_{mode}{cmb}.pdf")

    # Plot legend separately
    fname = "plots/legend_column.pdf"
    if not os.path.exists(fname):
        fig2, axes2 = plt.subplots()
        legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=5)
        fig2.canvas.draw()
        axes2.set_axis_off()
        bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
        plt.savefig(fname, bbox_inches=bbox)

# Plot total power output
fig, axes = plt.subplots(figsize=figsize)
ymax = 68
if mode == "both":
    axes.vlines(1, 0, ymax, "k")
    axes.vlines(1.5, 0, ymax, "k")
for i, config in enumerate(configs):
    colour = f"C{2 * i}"
    kw["color"] = colour
    power, time, energy, energy_time = get_data(config, modes, parsed_args)[:4]
    axes.plot(time, np.sum(power, axis=1), label=config.capitalize(), **kw)
    h = 3 if config == "aligned" else 6
    for t, e in zip(energy_time, energy):
        axes.annotate(f"{e:.2f} MWh", (t - 0.01, h), fontsize=10, color=colour)
axes.set_xticks(ticks)
axes.set_yticks(np.arange(10, ymax + eps, 10))
axes.set_xlim([ticks[0], ticks[-1]])
axes.set_ylim([0, ymax])
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"Power output [$\mathrm{MW}$]")
axes.legend(loc="upper left", handlelength=1.5, fontsize=14)
axes.grid(True)
plt.tight_layout()
config = "_".join(configs)
plot_dir = create_directory(f"plots/{config}/{approach}/{run}")
plt.savefig(f"{plot_dir}/{config}_total_power_output_{run}_{mode}.pdf")
