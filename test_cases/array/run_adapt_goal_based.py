from turbine_adapt import *
from turbine_adapt.adapt import GoalOrientedTidalFarm
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='test_cases/array/run_adapt_goal_based.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 0).""")
parser.add_argument('-ramp_level', 5, help="""
    Mesh resolution level inside the refined region for
    the ramp run.
    Choose a value from [0, 1, 2, 3, 4, 5] (default 5).""")
parser.add_argument('-num_tidal_cycles', 0.5)
parser.add_argument('-num_meshes', 40)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 5)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-target', 10000.0)
parser.add_argument('-h_min', 0.01)
parser.add_argument('-h_max', 100.0)
parser.add_argument('-turbine_h_max', 2.0)
parser.add_argument('-flux_form', False)
parser.add_argument('-approach', 'isotropic_dwr')
parser.add_argument('-error_indicator', 'difference_quotient')
parser.add_argument('-load_index', 0)
parser.add_argument('-staggered', False, help="""
    Toggle between aligned and staggered array
    (default False).""")
parsed_args = parser.parse_args()
if parsed_args.approach not in ('isotropic_dwr', 'anisotropic_dwr'):  # NOTE: anisotropic => weighted Hessian
    raise ValueError(f"Adaptation approach {parsed_args.approach} not recognised.")
if parsed_args.error_indicator != 'difference_quotient':
    raise NotImplementedError(f"Error indicator {parsed_args.error_indicator} not recognised.")
approach = parsed_args.approach.split('_')[0]

# Mesh independent setup
ramp_dir = os.path.join('outputs', 'fixed_mesh', f'level{parsed_args.ramp_level}')
options = ArrayOptions(
    level=parsed_args.level, staggered=parsed_args.staggered,
    ramp_level=parsed_args.ramp_level, ramp_dir=ramp_dir,
)
options.simulation_end_time = parsed_args.num_tidal_cycles*options.tide_time
root_dir = os.path.join(options.output_directory, approach)
options.output_directory = create_directory(os.path.join(root_dir, f'target{parsed_args.target:.0f}'))

# Run simulation
tidal_farm = GoalOrientedTidalFarm(options, root_dir, parsed_args.num_meshes)
tidal_farm.fixed_point_iteration(**parsed_args)
