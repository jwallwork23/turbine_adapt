from turbine_adapt import *
from firedrake_adjoint import *
from firedrake.adjoint.blocks import NonlinearVariationalSolveBlock as SolveBlock
from options import ArrayOptions
import h5py


# Parse arguments
parser = Parser(prog='turbine/array/run_goal_based.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-end_time', 8928.0)
parser.add_argument('-num_meshes', 160)
parser.add_argument('-maxiter', 5)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-target', 5000.0*400)
parser.add_argument('-h_min', 0.1)
parser.add_argument('-h_max', 1000)
parser.add_argument('-plot_metric', False)
args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'dwr', 'target{:.0f}'.format(args.target))
options.output_directory = create_directory(output_dir)
end_time = args.end_time
export_time = options.simulation_export_time
mesh_iteration_time = end_time/args.num_meshes
exports_per_mesh_iteration = int(mesh_iteration_time/export_time)
dt = options.timestep
timesteps = [dt for i in range(num_meshes)]  # TODO: adaptive timestepping

# Initial meshes are identical
meshes = args.num_meshes*[Mesh(options.mesh2d.coordinates)]
num_cells_old = args.num_meshes*[options.mesh2d.num_cells()]

# Enter fixed point iteration loop
converged = False
converged_reason = None
energy_old = None
t_epsilon = 1.0e-05
tape = get_working_tape()
for fp_iteration in range(args.maxiter + 1):
    outfiles = {'vtk': {}, 'hdf5': {}}
    if fp_iteration == args.maxiter:
        converged = True
        if converged_reason is None:
            converged_reason = 'maximum number of iterations reached'
    metrics = None if converged else [Function(TensorFunctionSpace(mesh, "CG", 1)) for mesh in meshes]

    # Solve forward on whole time interval, saving to hdf5
    tape.clear_tape()
    with stop_annotating():
        q = None
        for i, mesh in enumerate(meshes):

            # Re-initialise ArrayOptions object
            if not (fp_iteration == i == 0):
                options.__init__(mesh=mesh)
                options.create_tidal_farm()
            if not converged:
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

            # Set initial conditions for current mesh iteration, ensuring exports are not overwritten
            if i == 0:
                options.apply_initial_conditions(solver_obj)
                for f in options.fields_to_export:
                    outfiles['vtk'][f] = solver_obj.exporters['vtk'].exporters[f].outfile
                for f in options.fields_to_export_hdf5:
                    outfiles['hdf5'][f] = solver_obj.exporters['hdf5'].exporters[f].outfile
                outfiles['vorticity'] = cb._outfile
            else:
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
                for f in options.fields_to_export_hdf5:
                    solver_obj.exporters['hdf5'].exporters[f].outfile = outfiles['hdf5'][f]
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

    # Escape if converged
    if converged:
        print_output("Termination due to {:s}".format(converged_reason))
        break

    # Compute metrics on the subintervals in reverse, with no outputs
    adj_value_old = None
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

        def export_func():
            forward_solutions.append(solver_obj.fields.solution_2d.copy(deepcopy=True))
            solution_old = solver_obj.timestepper.solution_old
            forward_solutions_old.append(solution_old.copy(deepcopy=True))

        # Solve forward on current subinterval
        solver_obj.iterate(update_forcings=options.update_forcings, export_func=export_func)

        # Reverse mode propagation
        with stop_annotating():

            # Get seed for reverse mode propagation
            if i == args.num_meshes - 1:
                seed = energy
                adj_value = 1
            else:
                seed = solver_obj.fields.solution_2d
                adj_value = Function(solver_obj.function_spaces.V_2d)
                uv_adj, elev_adj = adj_value.split()
                uv_adj_old, elev_adj_old = adj_value_old.split()
                uv_adj.project(uv_adj_old)  # TODO: Use adjoint of project
                elev_adj.project(elev_adj_old)

            # Propagate through the reverse mode of AD
            adj_value = compute_gradient(seed, control, adj_value=adj_value)
            adj_value_old = adj_value

            # Extract adjoint solutions
            solve_blocks = [block for block in tape._blocks if isinstance(block, SolveBlock)]
            solve_blocks = [block for block in solve_blocks if block.adj_sol is not None]
            t = options.simulation_end_time
            next_export_t = options.simulation_end_time
            need_old = False
            adj_solutions = []
            adj_solutions_old = []
            iteration = -1
            while t > start_time + t_epsilon:
                if t <= next_export_t:
                    adj_solutions.append(solve_blocks[iteration].adj_sol)
                    need_old = True
                    next_export_t -= options.simulation_export_time
                if need_old:
                    adj_solutions_old.append(solve_blocks[iteration].adj_sol)
                    need_old = False
                t -= dt
                iteration -= 1

            """
            Accumulate metric on current subinterval.

            Note that we are using an endpoint quadrature rule, i.e. the quadrature node
            is at the end of the subinterval.
            """
            adjoint_solutions = list(reversed(adj_solutions))
            adjoint_solutions_old = list(reversed(adj_solutions_old))
            tozip = (forward_solutions, forward_solutions_old, adjoint_solutions, adjoint_solutions_old)
            wq = options.simulation_export_time
            for forward, forward_old, adjoint, adjoint_old in zip(*tozip):
                raise NotImplementedError  # TODO: (use difference quotients?)

    # Apply space-time normalisation
    if args.plot_metric:
        metric_file = File(os.path.join(options.output_directory, 'unnormalised_metric.pvd'))
        for metric in metrics:
            metric.rename('Normalised metric')
            metric_file._topology = None
            metric_file.write(metric)
    space_time_normalise(metrics, end_time, timesteps, args.target, args.norm_order)
    enforce_element_contraints(metrics, args.h_min, args.h_max)
    if args.plot_metric:
        metric_file = File(os.path.join(options.output_directory, 'metric.pvd'))
        for metric in metrics:
            metric.rename('Normalised metric')
            metric_file._topology = None
            metric_file.write(metric)

    # Adapt meshes, checking for convergence of element count
    num_cells = []
    elements_converged = True
    for i, (metric, num_cells_old_i) in enumerate(zip(metrics, num_cells_old)):
        meshes[i] = Mesh(adapt(meshes[i], metric).coordinates)
        num_cells_i = meshes[i].num_cells()
        num_cells.append(num_cells_i)
        if np.abs(num_cells_i - num_cells_old_i) > args.element_rtol*num_cells_old_i:
            elements_converged = False
    if elements_converged:
        print_output("Mesh element count converged to rtol {:.2e}".format(args.element_rtol))
        converged = True
        converged_reason = 'converged element counts'

    # Check for convergence of energy output
    if energy_old is None:
        energy_old = energy
    elif np.abs(energy - energy_old) > qoi_rtol*energy_old:
        converged = True
        converged_reason = 'converged energy output'
