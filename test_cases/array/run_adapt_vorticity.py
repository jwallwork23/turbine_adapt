from turbine_adapt import *
import itertools
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/run_adapt_vorticity.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-end_time', 8928.0)
parser.add_argument('-num_meshes', 160)
parser.add_argument('-maxiter', 4)
parser.add_argument('-element_rtol', 0.01)
parser.add_argument('-norm_order', 10)
parser.add_argument('-target', 5000.0*400)
parser.add_argument('-h_min', 0.1)
parser.add_argument('-h_max', 1000)
parser.add_argument('-plot_metric', False)
parser.add_argument('-load_metric', False)
args = parser.parse_args()
if args.level != 0:
    raise NotImplementedError  # TODO

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'vorticity', 'target{:.0f}'.format(args.target))
options.output_directory = create_directory(output_dir)
end_time = args.end_time
export_time = options.simulation_export_time
mesh_iteration_time = end_time/args.num_meshes
exports_per_mesh_iteration = int(mesh_iteration_time/export_time)

# Initial meshes are identical
meshes = args.num_meshes*[Mesh(options.mesh2d.coordinates)]
num_cells_old = args.num_meshes*[options.mesh2d.num_cells()]

# Enter fixed point iteration loop
converged = False
converged_reason = None
t_epsilon = 1.0e-05
di = os.path.join('outputs', 'vorticity')
fname = 'metric_{:d}_level{:d}'
stashed_metric = os.path.join(di, fname)
initial_metric = os.path.join(di, fname if args.load_metric else 'initial_' + fname)
stashed_mesh = os.path.join(di, 'mesh_{:d}_level{:d}')
for fp_iteration in range(args.maxiter + 1):
    if fp_iteration == args.maxiter:
        converged = True
        if converged_reason is None:
            converged_reason = 'maximum number of iterations reached'
    metrics = None if converged else [Function(TensorFunctionSpace(mesh, "CG", 1)) for mesh in meshes]

    # Solve over the subintervals in sequence
    q = None
    timesteps = []
    continued = False
    for i, mesh in enumerate(meshes):

        # Load metrics from previous simulation
        if fp_iteration == 0 and not continued:
            try:
                if args.load_metric:
                    plex = PETSc.DMPlex().create()
                    plex.createFromFile(stashed_mesh.format(i, args.level) + '.h5')
                    meshes[i] = Mesh(plex)
                with DumbCheckpoint(initial_metric.format(i, args.level), mode=FILE_READ) as chk:
                    chk.load(metrics[i], name="Metric")
                print_output("Using stashed metric for iteration {:d}.".format(i))
                timesteps.append(options.timestep)
                continue
            except Exception:
                print_output("Cannot load stashed metric for iteration {:d}.".format(i))
                if not os.path.exists(initial_metric.format(i, args.level) + '.h5'):
                    print_output("File does not exist.")

        # Re-initialise ArrayOptions object
        msg = "\n Fixed point iteration {:2d}: subinterval {:3d} \n".format(fp_iteration, i)
        print_output(43*'*' + msg + 43*'*')
        if not (fp_iteration == i == 0):
            options.__init__(mesh=mesh)
            options.create_tidal_farm()
        start_time = i*mesh_iteration_time
        options.simulation_end_time = (i+1)*mesh_iteration_time
        i_export = i*exports_per_mesh_iteration
        timesteps.append(options.timestep)
        options.no_exports = not converged
        if not converged:
            options.fields_to_export = []

        # Create a new solver object and assign boundary conditions
        solver_obj = FarmSolver(options)
        options.apply_boundary_conditions(solver_obj)

        # Create callbacks, ensuring data from previous meshes are not overwritten
        cb = PowerOutputCallback(solver_obj)
        cb._create_new_file = i == 0
        solver_obj.add_callback(cb, 'timestep')
        cb = PeakVorticityCallback(solver_obj, plot=converged)
        cb._create_new_file = i == 0
        solver_obj.add_callback(cb, 'timestep')

        # Set initial conditions for current mesh iteration, ensuring exports are not overwritten
        if i == 0:
            options.apply_initial_conditions(solver_obj)
        else:
            solver_obj.create_exporters()
            assert q is not None
            uv, elev = q.split()
            solver_obj.assign_initial_conditions(uv=uv, elev=elev)
            solver_obj.i_export = i_export
            solver_obj.next_export_t = i_export*export_time
            solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/options.timestep))
            solver_obj.simulation_time = solver_obj.iteration*options.timestep
            solver_obj.export_initial_state = False
            for f in options.fields_to_export:
                solver_obj.exporters['vtk'].exporters[f].set_next_export_ix(i_export + 1)
            if converged:
                cb._outfile.counter = itertools.count(start=i_export + 1)

        def export_func():
            """
            Compute isotropic vorticity metric from
            recovered vorticity field and accumulate
            by L1 normalisation, i.e. time integration
            """
            if converged:
                return  # Do not accumulate on the final run
            zeta = solver_obj.callbacks['timestep']['vorticity'].zeta
            t = solver_obj.simulation_time
            wq = export_time
            if np.isclose(t, start_time):
                wq *= 0.5
            elif t > options.simulation_end_time - options.timestep - t_epsilon:
                wq *= 0.5
            metrics[i] += wq*isotropic_metric(zeta, target_space=metrics[i].function_space())

        # Solve forward on current subinterval, extracting vorticity metrics
        solver_obj.iterate(update_forcings=options.update_forcings, export_func=export_func)
        q = solver_obj.fields.solution_2d.copy(deepcopy=True)

        # Save metric data to file during first fixed point iteration
        if not converged:
            fname = initial_metric if fp_iteration == 0 else stashed_metric
            with DumbCheckpoint(fname.format(i, args.level), mode=FILE_CREATE) as chk:
                chk.store(metrics[i], name="Metric")
            viewer = PETSc.Viewer().createHDF5(stashed_mesh.format(i, args.level), 'w')
            viewer(meshes[i].topology_dm)
    assert np.isclose(options.simulation_end_time, end_time)

    # Escape if converged
    if converged:
        print_output("Termination due to {:s}".format(converged_reason))
        break

    # Apply space-time normalisation
    if args.plot_metric:
        metric_file = File(os.path.join(options.output_directory, 'unnormalised_metric.pvd'))
        for metric in metrics:
            metric.rename('Unnormalised metric')
            metric_file._topology = None
            metric_file.write(metric)
        metric_file.close()
    space_time_normalise(metrics, end_time, timesteps, args.target, args.norm_order)
    enforce_element_constraints(metrics, args.h_min, args.h_max)
    if args.plot_metric:
        metric_file = File(os.path.join(options.output_directory, 'metric.pvd'))
        for metric in metrics:
            metric.rename('Normalised metric')
            metric_file._topology = None
            metric_file.write(metric)
        metric_file.close()

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