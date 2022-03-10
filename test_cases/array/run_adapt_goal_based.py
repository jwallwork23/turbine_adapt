from turbine_adapt import *
from turbine_adapt.adapt import GoalOrientedTidalFarm
from options import ArrayOptions


# Parse arguments
parser = Parser("test_cases/array/run_adapt_goal_based.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.parse_setup()
parser.parse_convergence_criteria()
parser.parse_metric_parameters()
parser.parse_indicator()
parser.parse_approach()
parser.parse_loading()
parsed_args = parser.parse_args()
approach = parsed_args.approach.split("_")[0]

# Mesh independent setup
ramp_dir = os.path.join("outputs", "fixed_mesh", f"level{parsed_args.ramp_level}")
options = ArrayOptions(
    level=parsed_args.level,
    staggered=parsed_args.config == "staggered",
    ramp_level=parsed_args.ramp_level,
    ramp_dir=ramp_dir,
)
options.simulation_end_time = (
    parsed_args.end_time or parsed_args.num_tidal_cycles * options.tide_time
)
root_dir = os.path.join(options.output_directory, approach)
options.output_directory = create_directory(
    os.path.join(root_dir, f"target{parsed_args.target:.0f}")
)

# Run simulation
tidal_farm = GoalOrientedTidalFarm(options, root_dir, parsed_args.num_meshes)
tidal_farm.fixed_point_iteration(**parsed_args)
