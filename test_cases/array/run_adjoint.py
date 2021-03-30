from turbine_adapt import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import NonlinearVariationalSolveBlock as SolveBlock
from options import ArrayOptions
import h5py


# Parse arguments
parser = Parser(prog='turbine/array/run_adjoint.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-end_time', 8928.0/40)
parser.add_argument('-num_meshes', 80//40)
args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'adjoint')
options.output_directory = create_directory(output_dir)
end_time = args.end_time
export_time = options.simulation_export_time
mesh_iteration_time = end_time/args.num_meshes
exports_per_mesh_iteration = int(mesh_iteration_time/export_time)
dt = options.timestep

# Initial meshes are identical
meshes = args.num_meshes*[Mesh(options.mesh2d.coordinates)]
num_cells_old = args.num_meshes*[options.mesh2d.num_cells()]

# Enter fixed point iteration loop
t_epsilon = 1.0e-05
tape = get_working_tape()
outfiles = {'vtk': {}}

# Solve forward on whole time interval, saving to hdf5
tape.clear_tape()
with stop_annotating():
    q = None
    for i, mesh in enumerate(meshes):

        # Re-initialise ArrayOptions object
        if i > 0:
            options.__init__(mesh=mesh)
            options.create_tidal_farm()
        options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']
        start_time = i*mesh_iteration_time
        options.simulation_end_time = (i+1)*mesh_iteration_time
        i_export = i*exports_per_mesh_iteration

        # Create a new solver object and assign boundary conditions
        solver_obj = FarmSolver(options)
        options.apply_boundary_conditions(solver_obj)

        # Create callbacks, ensuring data from previous meshes are not overwritten
        for Callback in (PowerOutputCallback, PeakVorticityCallback):
            cb = Callback(solver_obj)
            cb._create_new_file = i == 0
            solver_obj.add_callback(cb, 'timestep')

        solver_obj.create_exporters()

        # Set initial conditions for current mesh iteration, ensuring exports are not overwritten
        if i == 0:
            options.apply_initial_conditions(solver_obj)
            for f in options.fields_to_export:
                outfiles['vtk'][f] = solver_obj.exporters['vtk'].exporters[f].outfile
            outfiles['vorticity'] = cb._outfile
        else:
            solver_obj.create_exporters(outfiles=outfiles['vtk'])
            assert q is not None
            uv, elev = q.split()
            solver_obj.assign_initial_conditions(uv=uv, elev=elev)
            solver_obj.i_export = i_export
            solver_obj.next_export_t = i_export*export_time
            solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/dt))
            solver_obj.simulation_time = solver_obj.iteration*dt
            for f in options.fields_to_export:
                solver_obj.exporters['vtk'].exporters[f].outfile = outfiles['vtk'][f]
                solver_obj.exporters['vtk'].exporters[f].outfile._topology = None
            cb._outfile = outfiles['vorticity']

        # Solve forward on current subinterval
        solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)
        q = solver_obj.fields.solution_2d.copy(deepcopy=True)

# Evaluate energy output in megawatt hours
with h5py.File(os.path.join(output_dir, 'diagnostic_turbine.hdf5'), 'r') as f:
    power = np.array(f['current_power'])
total_power = np.sum(power, axis=1)*1030.0/1.0e+06/3600
energy = dt*(0.5*(total_power[0] + total_power[-1]) + np.sum(total_power[1:-1]))
print_output("Energy output: {:.4f} MW h".format(energy))

# Solve adjoint on the subintervals in reverse
adj_value_old = None
di = options.output_directory
fname = 'AdjointVelocity2d'
z_file = File(os.path.join(create_directory(os.path.join(di, fname)), fname + '.pvd'))
fname = 'AdjointElevation2d'
zeta_file = File(os.path.join(create_directory(os.path.join(di, fname)), fname + '.pvd'))
for i in range(args.num_meshes-1, -1, -1):
    mesh = meshes[i]
    tape.clear_tape()
    forward_solutions = []
    forward_solutions_old = []

    # Re-initialise ArrayOptions object
    options.__init__(mesh=mesh)
    options.create_tidal_farm()
    start_time = i*mesh_iteration_time
    options.simulation_end_time = (i+1)*mesh_iteration_time
    i_export = i*exports_per_mesh_iteration
    viscosity = Control(options.horizontal_viscosity)

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
    options.apply_boundary_conditions(solver_obj)

    # Set initial conditions for current mesh iteration
    if i == 0:
        options.apply_initial_conditions(solver_obj)
        solver_obj.export_initial_state = np.isclose(dt, options.simulation_export_time)
    else:
        solver_obj.load_state(i_export)
    control = Control(solver_obj.fields.solution_2d)
    options._energy = 0
    dtc = Constant(options.timestep)
    # u = split(solver_obj.fields.solution_2d)[0]
    u = solver_obj.fields.solution_2d.split()[0]

    def update_forcings(t):
        options._energy += assemble(dtc*sqrt(dot(u, u))*dot(u, u)*dx)  # TODO properly

    def export_func():
        forward_solutions.append(solver_obj.fields.solution_2d.copy(deepcopy=True))
        solution_old = solver_obj.timestepper.solution_old
        forward_solutions_old.append(solution_old.copy(deepcopy=True))

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=export_func)

    # Reverse mode propagation
    with stop_annotating():

        # Get seed for reverse mode propagation
        if i == args.num_meshes - 1:
            adj_value = 1
        else:
            adj_value = Function(solver_obj.function_spaces.V_2d)
            uv_adj, elev_adj = adj_value.split()
            uv_adj_old, elev_adj_old = adj_value_old.split()
            uv_adj.project(uv_adj_old)  # TODO: Use adjoint of project
            elev_adj.project(elev_adj_old)

        # Propagate through the reverse mode of AD
        # TODO: How to simultaneously take account of both energy and adj_value?
        print_output("Computing gradient...")
        adj_value = compute_gradient(options._energy, control, adj_value=adj_value)

        # Extract adjoint solutions
        solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
        solve_blocks = [block for block in solve_blocks if block.adj_sol is not None]
        t = options.simulation_end_time
        next_export_t = options.simulation_end_time
        adj_solutions = []
        adj_solutions_old = []
        iteration = -1
        while t > start_time + t_epsilon:
            if t <= next_export_t:
                z, zeta = solve_blocks[iteration].adj_sol.split()
                msg = "t = {:7.2f} z norm {:10.4f} zeta norm {:10.4f}"
                print_output(msg.format(t, norm(z), norm(zeta)))
                z.rename('Adjoint velocity')
                zeta.rename('Adjoint elevation')
                z_file.write(z)
                zeta_file.write(zeta)
                next_export_t -= options.simulation_export_time
            t -= dt
            iteration -= 1
        adj_value_old = control.adj_value
