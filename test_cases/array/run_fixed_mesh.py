from turbine_adapt import *
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/run_fixed_mesh.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', 'level{:d}'.format(args.level))
options.output_directory = create_directory(output_dir)

# Solve
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
for Callback in (PowerOutputCallback, PeakVorticityCallback):
    solver_obj.add_callback(Callback(solver_obj), 'timestep')
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
