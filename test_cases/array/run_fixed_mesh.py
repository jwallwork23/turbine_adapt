from turbine import *
from options import ArrayOptions
import pyadjoint


pyadjoint.pause_annotation()

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
parser.add_argument(
    "--load_index",
    help="Optional index to load from HDF5",
    type=int,
    default=0,
)
parser.parse_setup()
parsed_args = parser.parse_args()
config = parsed_args.configuration
load_index = parsed_args.load_index
approach = "uniform_mesh" if parsed_args.uniform else "fixed_mesh"

# Set parameters
nproc = COMM_WORLD.size
ramp_level = parsed_args.ramp_level
ramp_dir = f"outputs/{config}/{approach}/level{ramp_level}/ramp{nproc}/hdf5"
options = ArrayOptions(
    level=parsed_args.level,
    uniform=parsed_args.uniform,
    configuration=config,
    ramp_dir=ramp_dir,
    ramp_level=ramp_level,
    fields_to_export=["uv_2d"],
    fields_to_export_hdf5=["uv_2d", "elev_2d"],
)
options.simulation_end_time = (
    parsed_args.end_time or parsed_args.num_tidal_cycles * options.tide_time
)
options.create_tidal_farm()
output_dir = f"{options.output_directory}/{config}/{approach}/level{parsed_args.level}"
options.output_directory = create_directory(output_dir)

# Create solver
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
cb = PowerOutputCallback(solver_obj)
cb._create_new_file = load_index == 0
solver_obj.add_callback(cb, "timestep")
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
if load_index == 0:
    options.apply_initial_conditions(solver_obj)
else:
    idx = parsed_args.load_index
    print_output(f"Loading state at time {idx * options.simulation_export_time}")
    solver_obj.load_state(idx)

# Solve
solver_obj.iterate(
    update_forcings=options.update_forcings, export_func=options.export_func
)
