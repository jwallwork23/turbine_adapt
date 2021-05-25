from turbine_adapt import *
from pyadjoint import stop_annotating
from options import ArrayOptions


stop_annotating()

# Parse arguments
parser = Parser(prog='turbine/array/run_fixed_mesh.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-ramp_level', 5, help="""
    Mesh resolution level inside the refined region for
    the ramp run.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 5).""")
parser.add_argument('-num_tidal_cycles', 0.5)
parser.add_argument('-staggered', False, help="""
    Toggle between aligned and staggered array
    (default False).""")
parsed_args = parser.parse_args()

# Set parameters
ramp_level = parsed_args.ramp_level
ramp_dir = os.path.join('outputs', 'fixed_mesh', f'level{ramp_level}')
options = ArrayOptions(
    level=parsed_args.level, staggered=parsed_args.staggered,
    ramp_dir=ramp_dir, ramp_level=ramp_level,
)
options.simulation_end_time = parsed_args.num_tidal_cycles*options.tide_time
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', f'level{parsed_args.level}')
options.output_directory = create_directory(output_dir)
options.fields_to_export = ['uv_2d', 'elev_2d', 'vorticity_2d']

# Create solver
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
solver_obj.add_callback(PowerOutputCallback(solver_obj), 'timestep')
vorticity_2d = Function(solver_obj.function_spaces.P1_2d, name='vorticity_2d')
vorticity_calculator = VorticityCalculator2D(vorticity_2d, solver_obj)
solver_obj.add_new_field(
    vorticity_2d, 'vorticity_2d', 'Vorticity', 'Vorticity2d',
    unit='s-1', preproc_func=vorticity_calculator.solve,
)
options.apply_initial_conditions(solver_obj)

# Solve
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
