from thetis import create_directory
from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
from utils import *


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
parser.add_argument(
    "-c",
    "--columns",
    nargs="+",
    help="Turbine columns to use in QoI",
    default=[0, 1, 2, 3, 4],
)
parser.parse_setup()
parser.parse_approach(default="fixed_mesh")
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
configs = parsed_args.configurations
if len(configs) == 0:
    print("Nothing to plot.")
    sys.exit(0)
cols = np.sort([int(c) for c in parsed_args.columns])
ext = "".join([str(c) for c in cols])
if ext != "01234":
    parsed_args.approach += "_" + ext
approach = parsed_args.approach
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
    if len(ext) == 5:
        l = "fixed" if approach == "fixed_mesh" else approach.split("_dwr")[0]
        l = ["\n".join([config.capitalize(), f"({l})"])]
    elif len(ext) == 1:
        l = [f"Column {int(ext)+1}"]
    else:
        cols = ", ".join([f"{int(e) + 1}" for e in ext])
        l = [f"Columns {cols}"]
    axes.legend([whiteline], l, loc="upper right", handlelength=0, handletextpad=0, fontsize=18)
    plt.tight_layout()
    cmb = "_combined" if parsed_args.combine_plots else ""
    plot_dir = create_directory(f"plots/{config}/{approach}/{run}")
    plt.savefig(f"{plot_dir}/{config}_{approach}_power_output_column_{run}_{mode}{cmb}.pdf")

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
eps = 1.0e-05
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
plt.savefig(f"{plot_dir}/{config}_{approach}_total_power_output_{run}_{mode}.pdf")
