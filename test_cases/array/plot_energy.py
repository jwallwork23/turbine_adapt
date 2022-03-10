from thetis import create_directory
from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


# Parse arguments
parser = Parser("test_cases/array/plot_energy.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parsed_args = parser.parse_args()
config = parsed_args.configuration

# Setup directories
columns = [0, 1, 2, 3, 4, "overall"]
targets = 5000.0 * 2.0 ** np.array([0, 1, 2, 3, 4, 5])
cwd = os.path.dirname(__file__)
plot_dir = create_directory(os.path.join(cwd, "plots", config))

# Loop over runs
approaches = ["fixed_mesh", "isotropic"]
data = {
    approach: {"E": {column: [] for column in columns}, "dofs": []}
    for approach in approaches
}
for level, target in enumerate(targets):
    for approach in approaches:
        if approach == "fixed_mesh":
            run = f"level{level}"
        else:
            run = f"target{target:.0f}"
        input_dir = os.path.join(cwd, "outputs", config, approach, run)

        # Read DoF count from log
        logfile = os.path.join(input_dir, "log")
        if not os.path.exists(logfile):
            print(f"Cannot load logfile {level} for {approach}.")
            continue
        dofs = 0
        with open(logfile, "r") as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            dofs += int(f.readline().split()[-1])
            dofs += int(f.readline().split()[-1])
        data[approach]["dofs"].append(dofs)

        # Load data
        input_file = os.path.join(input_dir, "diagnostic_turbine.hdf5")
        if not os.path.exists(input_file):
            print(f"Cannot load dataset {level} for {approach}.")
            continue
        with h5py.File(input_file, "r") as f:
            power = np.array(f["current_power"])
            time = np.array(f["time"])
        assert len(power) == len(time)

        # Compute energy output using trapezium rule on each timestep
        for column in columns:
            if column == "overall":
                E = sum(data[approach]["E"][c][-1] for c in range(5))
                data[approach]["E"][column].append(E)
            else:
                indices = [3 * column, 3 * column + 1, 3 * column + 2]
                total_power = np.sum(power[:, indices], axis=1) * 1030.0 / 1.0e06
                energy = 0
                for i in range(len(total_power) - 1):
                    energy += (
                        0.5
                        * (time[i + 1] - time[i])
                        * (total_power[i + 1] + total_power[i])
                    )
                data[approach]["E"][column].append(energy / 3600)

# Plot
for column in columns:
    fig, axes = plt.subplots(figsize=(7, 6))
    for approach in approaches:
        label = approach.capitalize().replace("_", " ")
        axes.semilogx(
            data[approach]["dofs"], data[approach]["E"][column], "--x", label=label
        )
    axes.set_xlabel(r"DoF count")
    axes.set_ylabel(r"Energy [$\mathrm{MW\,h}$]")
    axes.legend()
    axes.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"energy_output_{column}.pdf"))
