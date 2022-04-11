from turbine_adapt import *
from turbine_adapt.plotting import *
from utils import *


# Parse user input
parser = Parser("test_cases/array/plot_energy.py")
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
loops = {
    "fixed_mesh": range(4),
    "isotropic_dwr": 5000.0 * 2.0 ** np.array(range(2)),  # TODO: 4 levels
    # "anisotropic_dwr": 2500.0 * 2.0 ** np.array(range(4)),  # TODO
}
if mode != "run":
    loops.pop("isotropic_dwr")
for approach, levels in loops.items():
    parsed_args.approach = approach
    for level in levels:

        # Count DoFs
        if approach == "fixed_mesh":
            parsed_args.level = level
            options = ArrayOptions(level=level, configuration=config)
            dofs = 9 * options.mesh2d.num_cells()  # NOTE: assumes P1DG-P1DG
        else:
            parsed_args.target_complexity = level
            cells = count_cells(f"outputs/{config}/{approach}/target{level:.0f}")
            dofs = 9 * np.mean(cells)  # NOTE: assumes P1DG-P1DG

        # Load data
        power, time = get_data(config, modes, parsed_args)[:2]

        # Compute energy output
        energy = time_integrate(power, time)
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
    axes.set_xlabel(r"Mean DoF count")
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
    axes.set_xlabel(r"Mean DoF count")
    axes.set_ylabel(r"Energy [$\mathrm{MW\,h}$]")
    axes.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{config}_energy_output_{approach}_01234_{mode}.pdf")
