from turbine_adapt import *
from turbine_adapt.plotting import *
from utils import time_integrate
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
parser.add_argument(
    "mode",
    help="Should we use ramp data and/or subsequent run?",
    choices=["run", "ramp", "both"],
    default="run",
)
parser.parse_setup()
parsed_args = parser.parse_args()
config = parsed_args.configuration
mode = parsed_args.mode
modes = ["ramp", "run"] if mode == "both" else [mode]
end_time = parsed_args.end_time
num_tidal_cycles = parsed_args.num_tidal_cycles

# Collect power/energy output data
energy_output = {0: {}, 1: {}, 2: {}, 3: {}, 4: {}, "overall": {}}
# loops = {"fixed_mesh": range(5), "isotropic": 5000.0 * 2.0 ** np.array(range(6))}
loops = {"fixed_mesh": range(5)}
for approach, levels in loops.items():
    for level in range(5):
        if approach == "fixed_mesh":
            options = ArrayOptions(level=level, configuration=config)
            if options.element_family == "dg-dg":
                dofs = 9 * options.mesh2d.num_cells()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError  # TODO: Read from log?
        output_dir = f"outputs/{config}/{approach}/level{level}"
        power = np.array([]).reshape((0, 15))
        time = np.array([]).reshape((0, 1))
        for m in modes:
            if end_time is None:
                end_time = 0.0
                if m != "run":
                    end_time += options.ramp_time
                if m != "ramp":
                    end_time += num_tidal_cycles * options.tide_time
            input_dir = output_dir + "/ramp" if m == "ramp" else output_dir
            fname = f"{input_dir}/diagnostic_turbine.hdf5"
            if not os.path.exists(fname):
                print(f"{fname} does not exist")
                continue
            with h5py.File(fname, "r") as h5:
                power = np.concatenate((power, np.array(h5["current_power"])))
                time = np.concatenate((time, np.array(h5["time"])))
        if len(time.flatten()) == 0:
            continue
        power = power[time.flatten() <= end_time, :]
        time = time[time <= end_time]
        power *= 1030.0 / 1.0e06  # MW
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

# Plot DoF count vs energy output
plot_dir = create_directory(f"plots/{config}")
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
    plt.savefig(f"{plot_dir}/{config}_energy_output_{subset}_{mode}.pdf")
