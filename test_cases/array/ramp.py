from turbine_adapt import *
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/ramp.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parsed_args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=parsed_args.level)
options.simulation_end_time = options.ramp_time
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', f'level{parsed_args.level}')
options.output_directory = create_directory(output_dir)

# Solve
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
# for Callback in (PowerOutputCallback, PeakVorticityCallback):
#     solver_obj.add_callback(Callback(solver_obj), 'timestep')
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)

# Store to checkpoint
uv, elev = solver_obj.fields.solution_2d.split()
with DumbCheckpoint(os.path.join(output_dir, 'ramp'), mode=FILE_CREATE) as chk:
    chk.store(uv)
    chk.store(elev)
