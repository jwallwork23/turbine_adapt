from turbine import *
from options import ArrayOptions
import pyadjoint


pyadjoint.pause_annotation()

# Parse arguments
parser = Parser("test_cases/array/ramp.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.parse_setup()
parser.add_argument(
    "--plot_vorticity",
    help="Plot fluid vorticity",
    action="store_true",
)
parser.add_argument(
    "--load_index",
    help="Optional index to load from HDF5",
    type=int,
    default=None,
)
parsed_args = parser.parse_args()
config = parsed_args.configuration
load_index = parsed_args.load_index
approach = "uniform_mesh" if parsed_args.uniform else "fixed_mesh"

# Set parameters
options = ArrayOptions(
    level=parsed_args.level,
    uniform=parsed_args.uniform,
    configuration=config,
    fields_to_export=[],
    fields_to_export_hdf5=["uv_2d", "elev_2d"],
    spunup=False,
)
options.simulation_end_time = options.ramp_time
options.create_tidal_farm()
output_dir = f"{options.output_directory}/{config}/{approach}/level{parsed_args.level}/ramp"
options.output_directory = create_directory(output_dir)

# Setup solver
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
if load_index is None:
    options.apply_initial_conditions(solver_obj)
else:
    idx = parsed_args.load_index
    print_output(f"Loading state at time {idx * options.simulation_export_time}")
    solver_obj.load_state(idx)

# Time integrate
solver_obj.iterate(
    update_forcings=options.update_forcings, export_func=options.export_func
)
