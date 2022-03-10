from turbine import *
from options import ArrayOptions


# Parse arguments
parser = Parser(prog="test_cases/array/ramp.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.parse_setup()
parsed_args = parser.parse_args()
config = parsed_args.configuration

# Set parameters
options = ArrayOptions(level=parsed_args.level, configuration=config)
options.simulation_end_time = options.ramp_time
options.create_tidal_farm()
output_dir = os.path.join(
    options.output_directory, config, "fixed_mesh", f"level{parsed_args.level}", "ramp"
)
options.output_directory = create_directory(output_dir)

# Solve
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
solver_obj.add_callback(PowerOutputCallback(solver_obj), "timestep")
vorticity_2d = Function(solver_obj.function_spaces.P1_2d, name="vorticity_2d")
uv_2d = solver_obj.fields.uv_2d
vorticity_calculator = VorticityCalculator2D(uv_2d, vorticity_2d)
solver_obj.add_new_field(
    vorticity_2d,
    "vorticity_2d",
    "Vorticity",
    "Vorticity2d",
    unit="s-1",
    preproc_func=vorticity_calculator.solve,
)
options.fields_to_export = ["uv_2d", "elev_2d", "vorticity_2d"]
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(
    update_forcings=options.update_forcings, export_func=options.export_func
)

# Store to checkpoint
uv, elev = solver_obj.fields.solution_2d.split()
with DumbCheckpoint(os.path.join(output_dir, "ramp"), mode=FILE_CREATE) as chk:
    chk.store(uv)
    chk.store(elev)
