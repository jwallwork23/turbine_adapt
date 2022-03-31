from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *  # noqa
from options import ArrayOptions
import meshio
import matplotlib.pyplot as plt
import numpy as np


# Parse user input
parser = Parser("test_cases/array/plot_dofs.py")
parser.parse_setup()
parser.parse_approach()
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
num_meshes = parsed_args.num_meshes
approach = parsed_args.approach
if approach == "fixed_mesh":
    run = f"level{parsed_args.level}"
else:
    run = f"target{parsed_args.target_complexity:.0f}"
options = ArrayOptions(meshgen=True)

# Read cell data from VTU
dofs = {}
for config in ("aligned", "staggered"):
    cells = []
    for i in range(40):
        mesh = meshio.read(f"outputs/{config}/{approach}/{run}/Mesh2d_{i}.vtu")
        for cell_block in mesh.cells:
            if cell_block.type in ("triangle"):
                num_cells = len(cell_block)
                print(f"{i:2d}: {num_cells:6d} elements, {len(mesh.points):6d} vertices")
                cells.append(num_cells)
                continue
    dofs[config] = 9 * np.array(cells)
x = np.array(range(num_meshes)) + 0.5
ticks = np.arange(np.floor(x[0]), x[-1] + 1.0, 10)
times = (options.ramp_time + ticks / ticks[-1] * options.simulation_end_time) / options.tide_time
ticklabels = [f"{t:.3f}" for t in times]

# Plot DoF counts as a bar chart
fig, axes = plt.subplots()
for i, (config, dof) in enumerate(dofs.items()):
    axes.bar(x + i * 0.5, dof / 1000, width=0.5, label=config.capitalize(), color=f"C{2 * i}")
axes.set_xticks(ticks)
axes.set_xticklabels(ticklabels)
axes.set_xlim([ticks[0], ticks[-1]])
axes.set_ylim([0, 900])
axes.set_xlabel(r"Time/$T_{\mathrm{tide}}$")
axes.set_ylabel(r"DoF count ($\times10^3$)")
axes.legend(fontsize=16)
axes.grid(True)
plt.tight_layout()
plt.savefig(f"plots/{approach}_{run}_dofs.pdf")
