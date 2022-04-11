from turbine_adapt import *
from turbine_adapt.parse import Parser, positive_float
from turbine_adapt.plotting import *
from firedrake import *
from options import ArrayOptions


# Parse arguments
parser = Parser("test_cases/array/plot_mesh_reynolds_number.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.parse_approach()
parsed_args = parser.parse_args()
config = parsed_args.configuration
uniform = parsed_args.pop("approach") == "uniform_mesh"

# Set parameters
options = ArrayOptions(uniform=uniform, **parsed_args)

# Plot viscosity
nu = options.horizontal_viscosity
if isinstance(nu, Constant):
    print("Constant (kinematic) viscosity = {:.4e}".format(nu.values()[0]))
else:
    fig, axes = plt.subplots(figsize=(12, 6))
    levels = np.linspace(0.0, 1.1 * nu.dat.data.max(), 50)
    tc = tricontourf(nu, axes=axes, levels=levels, cmap="coolwarm")
    cbar = fig.colorbar(tc, ax=axes)
    cbar.set_label(r"(Kinematic) viscosity [$\mathrm m^2\,\mathrm s^{-1}$]")
plt.savefig(f"plots/{config}/viscosity.jpg")

# Plot mesh Reynolds number
fig, axes = plt.subplots(figsize=(12, 6))
Re_h = options.check_mesh_reynolds_number()
tc = tricontourf(Re_h, levels=50, axes=axes, cmap="coolwarm")
cbar = fig.colorbar(tc, ax=axes)
cbar.set_label("Mesh Reynolds number")
plt.tight_layout()
plt.savefig(f"plots/{config}/mesh_reynolds_number.jpg")
