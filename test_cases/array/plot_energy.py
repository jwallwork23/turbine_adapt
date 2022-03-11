from turbine_adapt import *
from turbine_adapt.plotting import *
from options import ArrayOptions
import h5py
import numpy as np


# Parse user input
parser = Parser("test_cases/array/plot_convergence.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parsed_args = parser.parse_args()
config = parsed_args.configuration


def time_integrate(arr, dt=2.232):
    """
    Time integrate an array of turbine power outputs
    that were obtained using Crank-Nicolson.

    :arg arr: the (n_timesteps, n_turbines) array
    :kwarg dt: the timestep
    """
    zeros = np.zeros(arr.shape[1])
    off1 = np.vstack((zeros, arr))
    off2 = np.vstack((arr, zeros))
    return dt * 0.5 * np.sum(off1 + off2, axis=0)


# Setup directories
if COMM_WORLD.size > 1:
    print_output(
        f"Will not attempt to plot with {COMM_WORLD.size} processors."
        " Run again in serial."
    )
    sys.exit(0)
pwd = os.path.dirname(__file__)
plot_dir = create_directory(os.path.join(pwd, "plots", config))
output_dir = create_directory(os.path.join(pwd, "outputs", config))

# Collect power/energy output data
energy_output = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, "overall": {}}
loops = {"fixed_mesh": range(5), "isotropic": 5000.0 * 2.0 ** np.array(range(6))}
for approach, levels in loops.items():
    for level in range(5):
        fname = f"{output_dir}/{approach}/level{level}/diagnostic_turbine.hdf5"
        if not os.path.exists(fname):
            print_output(f"{fname} does not exist")
            continue
        if approach == "fixed_mesh":
            options = ArrayOptions(level=level, staggered=config == "staggered")
            dofs = 9 * options.mesh2d.num_cells()  # NOTE: assumes dg-dg
        else:
            raise NotImplementedError  # TODO: Read from log
        with h5py.File(fname, "r") as h5:
            power = np.array(h5["current_power"]) * 1030.0 / 1.0e06  # MW
            energy = time_integrate(power) / 3600.0  # MWh
            for i in range(5):
                if approach not in energy_output[i]:
                    energy_output[i][approach] = {}
                j = 3 * i
                k = 3 * (i + 1)
                energy_output[i][approach][dofs] = sum(energy[j:k])
            if approach not in energy_output["overall"]:
                energy_output["overall"][approach] = {}
            energy_output["overall"][approach][dofs] = sum(energy)

# Plot DoF count vs energy output
for subset, byapproach in energy_output.items():
    fig, axes = plt.subplots()
    for label, bydof in byapproach.items():
        name = label.capitalize().replace("_", " ")
        dofs = list(bydof.keys())
        E = list(bydof.values())
        axes.plot(dofs, E, "-x", label=name)
    axes.set_xlabel(r"DoF count")
    axes.set_ylabel(r"Energy [$\mathrm{MW\,h}$]")
    axes.grid(True)
    axes.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/energy_output_{subset}.pdf")
