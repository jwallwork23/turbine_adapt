from turbine_adapt import *
from turbine_adapt.plotting import *
from options import ArrayOptions
import argparse
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-max_mesh_reynolds_number", help="""
    Maximum tolerated mesh Reynolds number (default 1000)""")
parser.add_argument("-base_viscosity", help="""
    Base viscosity (default 1)""")
parser.add_argument("-target_viscosity", help="""
    Target viscosity (default 0.01)""")
args = parser.parse_args()

# Set parameters
kwargs = {
    'base_viscosity': float(args.base_viscosity or 1.0),
    'target_viscosity': float(args.target_viscosity or 0.01),
    'max_mesh_reynolds_number': float(args.max_mesh_reynolds_number or 1000),
}
options = ArrayOptions(**kwargs)

# Plot viscosity
nu = options.horizontal_viscosity
if isinstance(nu, Constant):
    print("Constant (kinematic) viscosity = {:.4e}".format(nu.values()[0]))
else:
    fig, axes = plt.subplots(figsize=(12, 6))
    levels = np.linspace(0.9*options.target_viscosity, 1.1*options.base_viscosity, 50)
    tc = tricontourf(nu, axes=axes, levels=levels, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes)
    cbar.set_label(r"(Kinematic) viscosity [$\mathrm m^2\,\mathrm s^{-1}$]")
    cbar.set_ticks(np.linspace(0, options.base_viscosity, 5))
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
plt.savefig(os.path.join(plot_dir, 'viscosity.jpg'))

# Plot mesh Reynolds number
fig, axes = plt.subplots(figsize=(12, 6))
levels = np.linspace(0, 1.1*options.max_mesh_reynolds_number, 50)
Re_h, Re_h_min, Re_h_max = options.check_mesh_reynolds_number()
tc = tricontourf(Re_h, levels=levels, axes=axes, cmap='coolwarm')
cbar = fig.colorbar(tc, ax=axes)
cbar.set_label("Mesh Reynolds number")
cbar.set_ticks(np.linspace(0, options.max_mesh_reynolds_number, 5))
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'mesh_reynolds_number.jpg'))
