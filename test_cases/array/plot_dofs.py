from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *  # noqa
from options import ArrayOptions
from utils import count_cells
import matplotlib.pyplot as plt
import numpy as np


# Parse user input
parser = Parser("test_cases/array/plot_dofs.py")
parser.add_argument(
    "--columns",
    help="Compare turbine columns",
    action="store_true",
)
parser.parse_setup()
parser.parse_approach()
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
num_meshes = parsed_args.num_meshes
approach = parsed_args.approach
col_mode = parsed_args.columns
if approach == "fixed_mesh":
    run = f"level{parsed_args.level}"
else:
    run = f"target{parsed_args.target_complexity:.0f}"
options = ArrayOptions(meshgen=True)

# Read cell data from VTU
dofs = {}
if col_mode:
    for ext in ("", "_2", "_4"):
        fpath = f"outputs/staggered/{approach + ext}/{run}"
        cells = count_cells(fpath)
        dofs[approach + ext] = 9 * np.array(cells)
else:
    for config in ("aligned", "staggered"):
        fpath = f"outputs/{config}/{approach}/{run}"
        cells = count_cells(fpath)
        dofs[config] = 9 * np.array(cells)
for val in dofs.values():
    assert len(val) == parsed_args.num_meshes
x = np.array(range(num_meshes)) + 0.5
ticks = np.arange(np.floor(x[0]), x[-1] + 1.0, 10)
times = (options.ramp_time + ticks / ticks[-1] * options.simulation_end_time) / options.tide_time
ticklabels = [f"{t:.3f}" for t in times]

# Print statistics
for key, dof in dofs.items():
    min = np.min(dof)
    max = np.max(dof)
    mean = np.mean(dof)
    std = np.std(dof)
    total = np.sum(dof)
    print(f"{key:9s}: min   {min/9:.0f} elements / {min:.0f} DoFs")
    print(f"{key:9s}: max   {max/9:.0f} elements / {max:.0f} DoFs")
    print(f"{key:9s}: mean  {mean/9:.0f} elements / {mean:.0f} DoFs")
    print(f"{key:9s}: std   {std/9:.0f} elements / {std:.0f} DoFs")
    print(f"{key:9s}: total {total/9:.0f} elements / {total:.0f} DoFs")

# Plot DoF counts as a bar chart
fig, axes = plt.subplots()
n = len(dofs.keys())
for i, (key, dof) in enumerate(dofs.items()):
    if col_mode:
        c = key.split(approach)[-1]
        label = "Whole array" if c == "" else f"Column {int(c[-1])+1}"
    else:
        label = key.capitalize()
    axes.bar(x, dof / 1000, label=label, edgecolor=f"C{2 * i}", color="None")
axes.set_xticks(ticks)
axes.set_xticklabels(ticklabels)
axes.set_xlim([ticks[0], ticks[-1]])
axes.set_ylim([0, 900])
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"DoF count ($\times10^3$)")
axes.legend(fontsize=16)
axes.grid(True)
plt.tight_layout()
if col_mode:
    plt.savefig(f"plots/staggered_{approach}_{run}_dofs.pdf")
else:
    plt.savefig(f"plots/{approach}_{run}_dofs.pdf")
