from turbine_adapt import *
import itertools
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/run_fixed_mesh_subintervals.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-end_time', 8928.0)
parser.add_argument('-num_meshes', 1)
args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', 'level{:d}'.format(args.level))
options.output_directory = create_directory(output_dir)
end_time = args.end_time
export_time = options.simulation_export_time
mesh_iteration_time = end_time/args.num_meshes
exports_per_mesh_iteration = int(mesh_iteration_time/export_time)


def solver(ic, t_start, t_end, dt, J=0, qoi=None):
    options.simulation_end_time = t_end
    if np.isclose(t_end, end_time):
        options.simulation_end_time += 0.5*options.timestep
    i_export = i*exports_per_mesh_iteration

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
    options.apply_boundary_conditions(solver_obj)
    options.J = J

    # Create callbacks
    cb = PowerOutputCallback(solver_obj)
    cb._create_new_file = i == 0
    solver_obj.add_callback(cb, 'timestep')
    cb = PeakVorticityCallback(solver_obj)
    cb._create_new_file = i == 0
    solver_obj.add_callback(cb, 'timestep')

    # Set initial conditions for current mesh iteration
    if i == 0:
        options.apply_initial_conditions(solver_obj)
    else:
        solver_obj.create_exporters()
        uv, elev = ic.split()
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)
        solver_obj.i_export = i_export
        solver_obj.next_export_t = i_export*export_time
        solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/options.timestep))
        solver_obj.simulation_time = t_start
        solver_obj.export_initial_state = False
        for f in options.fields_to_export:
            solver_obj.exporters['vtk'].exporters[f].set_next_export_ix(i_export + 1)
        cb._outfile.counter = itertools.count(start=i_export + 1)  # FIXME

    def update_forcings(t):
        options.update_forcings(t)
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
    return solver_obj.fields.solution_2d.copy(deepcopy=True), options.J


# Solve over the subintervals in sequence
q = None
J = 0
for i in range(args.num_meshes):
    q, J = solver(q, i*mesh_iteration_time, (i+1)*mesh_iteration_time, options.timestep, J=J, qoi=None)
