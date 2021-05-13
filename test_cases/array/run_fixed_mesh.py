from turbine_adapt import *
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/ramp.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-num_tidal_cycles', 0.5)
parsed_args = parser.parse_args()

# Set parameters
level = parsed_args.level
options = ArrayOptions(level=level, ramp_dir=os.path.join('outputs', 'fixed_mesh', f'level{level}'))
options.simulation_end_time = parsed_args.num_tidal_cycles*options.tide_time
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', f'level{level}')
options.output_directory = create_directory(output_dir)

# Solve
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
solver_obj.add_callback(PowerOutputCallback(solver_obj), 'timestep')
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
