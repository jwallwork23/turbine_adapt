from turbine_adapt import *
from turbine_adapt.plotting import *
from utils import *


# TODO: use utils.get_data

# Parse user input
parser = Parser("test_cases/array/plot_convergence.py")
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
parser.parse_approach()
parser.parse_metric_parameters()
parser.parse_setup()
parsed_args = parser.parse_args()
config = parsed_args.configuration
mode = parsed_args.mode
modes = ["ramp", "run"] if mode == "both" else [mode]

# Collect power/energy output data
energy_output = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, "overall": {}}
# loops = {"fixed_mesh": range(5), "isotropic": 5000.0 * 2.0 ** np.array(range(6))}
loops = {"fixed_mesh": range(5)}
for approach, levels in loops.items():
    parsed_args.approach = approach
    for level in levels:
        parsed_args.level = level
        # parsed_args.target_complexity =
        if approach == "fixed_mesh":
            options = ArrayOptions(level=level, configuration=config)
            if options.element_family == "dg-dg":
                dofs = 9 * options.mesh2d.num_cells()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError  # TODO: Read from log?
        output_dir = f"outputs/{config}/{approach}/level{level}"
        power, time, energy, energy_time = get_data(config, modes, parsed_args)[:4]

        energy = time_integrate(power, time) / 3600.0  # MWh
        for i in range(5):
            if approach not in energy_output[i]:
                energy_output[i][approach] = {}
            j = 3 * i
            k = 3 * (i + 1)
            energy_output[i][approach][dofs] = sum(energy[j:k])
        if approach not in energy_output["overall"]:
            energy_output["overall"][approach] = {}
        energy_output["overall"][approach][dofs] = sum(energy)

# Plot formatting
colours = ["b", "C0", "mediumturquoise", "mediumseagreen", "g"]
kw = {"linewidth": 1.0}

# Plot DoF count vs energy output
plot_dir = create_directory(f"plots/{config}")
for subset, byapproach in energy_output.items():
    fig, axes = plt.subplots()
    for label, bydof in byapproach.items():
        name = label.capitalize().replace("_", " ")
        dofs = list(bydof.keys())
        E = list(bydof.values())
        axes.plot(dofs, E, "-x", label=name, **kw)
    axes.set_xlabel(r"DoF count")
    axes.set_ylabel(r"Energy [$\mathrm{MW\,h}$]")
    axes.grid(True)
    axes.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{config}_energy_output_{subset}_{mode}.pdf")

# Plot again by overlaid
for approach, levels in loops.items():
    fig, axes = plt.subplots()
    for subset, byapproach in energy_output.items():
        if subset == "overall":
            continue
        dofs = byapproach[approach].keys()
        E = byapproach[approach].values()
        kw["color"] = colours[subset]
        axes.plot(dofs, E, "-x", label=subset, **kw)
    axes.set_xlabel(r"DoF count")
    axes.set_ylabel(r"Energy [$\mathrm{MW\,h}$]")
    axes.grid(True)
    axes.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{config}_energy_output_{approach}_01234_{mode}.pdf")
