from turbine import *
from options import ArrayOptions


# Parse arguments
parser = Parser("test_cases/array/run_fixed_mesh.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.add_argument(
    "--plot_vorticity",
    help="Plot fluid vorticity",
    action="store_true",
)
parser.add_argument(
    "--use_direct_solver",
    help="Apply a full LU decomposition",
    action="store_true",
)
parser.parse_setup()
parsed_args = parser.parse_args()
config = parsed_args.configuration

# Set parameters
ramp_level = parsed_args.ramp_level
ramp_dir = os.path.join("outputs", config, "fixed_mesh", f"level{ramp_level}")
options = ArrayOptions(
    level=parsed_args.level,
    staggered=config == "staggered",
    ramp_dir=ramp_dir,
    ramp_level=ramp_level,
)
options.simulation_end_time = (
    parsed_args.end_time or parsed_args.num_tidal_cycles * options.tide_time
)
options.create_tidal_farm()
output_dir = os.path.join(
    options.output_directory, config, "fixed_mesh", f"level{parsed_args.level}"
)
options.output_directory = create_directory(output_dir)

# Create solver
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
solver_obj.add_callback(PowerOutputCallback(solver_obj), "timestep")
if parsed_args.plot_vorticity:
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
    options.fields_to_export.append("vorticity_2d")
if parsed_args.use_direct_solver:
    options.swe_timestepper_options.solver_parameters = {
        "mat_type": "aij",
        "snes_type": "newtonls",
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    }
options.apply_initial_conditions(solver_obj)

# Solve
solver_obj.iterate(
    update_forcings=options.update_forcings, export_func=options.export_func
)
