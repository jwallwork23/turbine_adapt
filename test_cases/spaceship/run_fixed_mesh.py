from turbine import *
from options import SpaceshipOptions


# Parse arguments
parser = Parser(prog='test_cases/spaceship/run_fixed_mesh.py')
parser.add_argument('-num_tidal_cycles', 3)
parsed_args = parser.parse_args()

# Set parameters
ramp_dir = os.path.join('outputs', 'fixed_mesh')
options = SpaceshipOptions(ramp_dir=ramp_dir)
options.simulation_end_time = parsed_args.num_tidal_cycles*24*3600.0
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh')
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
