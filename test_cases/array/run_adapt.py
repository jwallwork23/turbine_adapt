from thetis import print_output, COMM_WORLD
from turbine_adapt import *
from turbine_adapt.adapt import GoalOrientedTidalFarm
from options import ArrayOptions
import datetime


print_output(f"Start time: {datetime.datetime.now()}")

# Parse arguments
parser = Parser("test_cases/array/run_adapt.py")
parser.add_argument(
    "config",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.add_argument(
    "-c",
    "--columns",
    nargs="+",
    help="Turbine columns to use in QoI",
    default=[0, 1, 2, 3, 4],
)
parser.parse_setup()
parser.parse_convergence_criteria()
parser.parse_metric_parameters()
parser.parse_indicator()
parser.parse_approach()
parser.parse_loading()
parsed_args = parser.parse_args()
config = parsed_args.config
cols = np.sort([int(c) for c in parsed_args.columns])

# Mesh independent setup
nproc = COMM_WORLD.size
ramp_approach = parsed_args.ramp_approach
ramp_level = parsed_args.ramp_level
ramp_dir = f"outputs/{config}/{ramp_approach}/level{ramp_level}/ramp{nproc}/hdf5"
options = ArrayOptions(
    level=parsed_args.level,
    configuration=config,
    ramp_level=ramp_level,
    ramp_dir=ramp_dir,
)
num_cycles = parsed_args.num_tidal_cycles
options.simulation_end_time = parsed_args.end_time or num_cycles * options.tide_time

# Select columns to add to the QoI
approach = parsed_args.approach
target = parsed_args.target_complexity
qoi_farm_ids = tuple(np.vstack([options.column_ids[c] for c in cols]).flatten())
print_output(f"Using a QoI based on columns {cols}")
print_output(f"i.e. farm IDs {qoi_farm_ids}")
root_dir = f"{options.output_directory}/{config}/{approach}"
if len(cols) != 5:
    root_dir += "_" + "".join([str(c) for c in cols])
options.output_directory = create_directory(f"{root_dir}/target{target:.0f}")
print_output(f"Outputting to {options.output_directory}")

# Create farm and run simulation
num_meshes = parsed_args.num_meshes
tidal_farm = GoalOrientedTidalFarm(
    options, root_dir, num_meshes, qoi_farm_ids=qoi_farm_ids
)
tidal_farm.fixed_point_iteration(**parsed_args)
print_output(f"End time: {datetime.datetime.now()}")
