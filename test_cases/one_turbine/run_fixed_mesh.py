"""
2D flow around a tidal turbine
==============================

Based on the Thetis flow around a cylinder test case:
    thetis/examples/cylinder_eddies/cylinder_eddies.py.

Joe Wallwork, 08/02/2021.
"""
from turbine_adapt import *
from options import TurbineOptions
import sys


# Parse arguments
parser = Parser()
parser.add_argument('-plot_drag', False)
args = parser.parse_args()

# Set parameters
options = TurbineOptions()

# Plot drag field
if args.plot_drag:
    if COMM_WORLD.size > 1:
        msg = "Will not attempt to plot with {:d} processors. Run again in serial."
        print_output(msg.format(COMM_WORLD.size))
        sys.exit(0)
    from turbine_adapt.plotting import *
    fig, axes = plt.subplots(figsize=(9.5, 3.5))
    eps = 1.0e-05
    cd_max = options.quadratic_drag_coefficient.vector().gather().max()
    levels = np.linspace(-eps, cd_max + eps)
    tc = tricontourf(options.quadratic_drag_coefficient, axes=axes, cmap='coolwarm', levels=levels)
    cb = fig.colorbar(tc, ax=axes)
    cb.set_ticks(np.linspace(0.0, cd_max, 11))
    cb.set_label("Quadratic drag")
    fig.savefig(os.path.join(create_directory('plots'), 'drag.jpg'))
    sys.exit(0)

# Run
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
