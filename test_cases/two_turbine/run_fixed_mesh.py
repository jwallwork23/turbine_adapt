from turbine_adapt import *
from turbine_adapt.callback import PeakVorticityCallback, PowerOutputCallback
from options import TwoTurbineOptions


# Parse arguments
parser = Parser()
parser.add_argument('-level', 1, help="""
    Mesh resolution level inside the refined region
    Choose a value from [0, 1, 2, 3, 4] (default 1)""")
parser.add_argument('-viscosity', 0.5, help="""
    Horizontal viscosity (default 0.5)""")
parser.add_argument('-thrust', 0.8, help="""
    Thrust coefficient for the turbines (default 0.8)""")
parser.add_argument('-correct_thrust', True, help="""
    Apply thrust correction of [Kramer and Piggott 2016]? (default True)""")
args = parser.parse_args()

# Set parameters
kwargs = {
    'level': args.level,
    'box': True,
    'viscosity': args.viscosity,
    'thrust_coefficient': args.thrust,
    'correct_thrust': args.correct_thrust,
}
options = TwoTurbineOptions(**kwargs)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'level{:d}'.format(args.level))
options.output_directory = create_directory(output_dir)

# Setup solver
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
for Callback in (PowerOutputCallback, PeakVorticityCallback):
    solver_obj.add_callback(Callback(solver_obj), 'timestep')
options.apply_initial_conditions(solver_obj)

# Run
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
