from turbine_adapt import *
from turbine_adapt.adapt import GoalOrientedTidalFarm
from options import ArrayOptions


# Parse arguments
parser = Parser("test_cases/array/run_adapt_goal_based.py")
parser.add_argument(
    "config",
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
config = parsed_args.config
target = parsed_args.target_complexity
approach = parsed_args.approach
ramp_level = parsed_args.ramp_level

# Mesh independent setup
ramp_dir = f"outputs/{config}/fixed_mesh/level{ramp_level}/ramp"
options = ArrayOptions(
    level=parsed_args.level,
    configuration=config,
    ramp_level=ramp_level,
    ramp_dir=ramp_dir,
)
options.simulation_end_time = (
    parsed_args.end_time or parsed_args.num_tidal_cycles * options.tide_time
)
root_dir = f"{options.output_directory}/{config}/{approach}"
options.output_directory = create_directory(f"{root_dir}/target{target:.0f}")

# Run simulation
tidal_farm = GoalOrientedTidalFarm(options, root_dir, parsed_args.num_meshes)
tidal_farm.fixed_point_iteration(**vars(parsed_args))
