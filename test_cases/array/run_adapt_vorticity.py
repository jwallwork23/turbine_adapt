from turbine_adapt import *
import itertools
from options import ArrayOptions


pause_annotation()

# Parse user input
parser = Parser("test_cases/array/run_adapt_vorticity.py")
parser.parse_setup()
parser.parse_convergence_criteria()
parser.parse_metric_parameters()
parser.parse_loading()
args = parser.parse_args()

# Set parameters
options = ArrayOptions(level=args.level)
options.create_tidal_farm()
output_dir = os.path.join(
    options.output_directory, "vorticity", f"target{args.target:.0f}"
)
options.output_directory = create_directory(output_dir)
end_time = args.end_time
export_time = options.simulation_export_time
mesh_iteration_time = end_time / args.num_meshes
exports_per_mesh_iteration = int(mesh_iteration_time / export_time)

# Initial meshes are identical
meshes = args.num_meshes * [Mesh(options.mesh2d.coordinates)]
num_cells_old = args.num_meshes * [options.mesh2d.num_cells()]

# Enter fixed point iteration loop
converged = False
converged_reason = None
t_epsilon = 1.0e-05
di = os.path.join("outputs", "vorticity")
fname = "metric_{:d}_level{:d}"
stashed_metric = os.path.join(di, fname)
initial_metric = os.path.join(di, fname if args.load_metric else "initial_" + fname)
stashed_mesh = os.path.join(di, "mesh_{:d}_level{:d}")
for fp_iteration in range(args.maxiter + 1):
    if fp_iteration == args.maxiter:
        converged = True
        if converged_reason is None:
            converged_reason = "maximum number of iterations reached"
    metrics = (
        None
        if converged
        else [Function(TensorFunctionSpace(mesh, "CG", 1)) for mesh in meshes]
    )

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
                    plex.createFromFile(stashed_mesh.format(i, args.level) + ".h5")
                    meshes[i] = Mesh(plex)
                with DumbCheckpoint(
                    initial_metric.format(i, args.level), mode=FILE_READ
                ) as chk:
                    chk.load(metrics[i], name="Metric")
                print_output(f"Using stashed metric for iteration {i}.")
                timesteps.append(options.timestep)
                continue
            except Exception:
                print_output(f"Cannot load stashed metric for iteration {i}.")
                if not os.path.exists(initial_metric.format(i, args.level) + ".h5"):
                    print_output("File does not exist.")

        # Re-initialise ArrayOptions object
        print_output(
            43 * "*"
            + f"\n Fixed point iteration {fp_iteration:2d}: subinterval {i:3d} \n"
            + 43 * "*"
        )
        if not (fp_iteration == i == 0):
            options.__init__(mesh=mesh)
            options.create_tidal_farm()
        start_time = i * mesh_iteration_time
        options.simulation_end_time = (i + 1) * mesh_iteration_time
        i_export = i * exports_per_mesh_iteration
        timesteps.append(options.timestep)
        options.no_exports = not converged
        if not converged:
            options.fields_to_export = []

        # Create a new solver object and assign boundary conditions
        solver_obj = FarmSolver(options)
        options.apply_boundary_conditions(solver_obj)

        # Create power callback, ensuring data from previous meshes are not overwritten
        cb = PowerOutputCallback(solver_obj)
        cb._create_new_file = i == 0
        solver_obj.add_callback(cb, "timestep")

        # Create vorticity calculator
        if not options.no_exports:
            options.fields_to_export.append("vorticity_2d")
        vorticity_2d = Function(solver_obj.function_spaces.P1_2d, name="vorticity_2d")
        vorticity_calculator = VorticityCalculator2D(vorticity_2d, solver_obj)
        solver_obj.add_new_field(
            vorticity_2d,
            "vorticity_2d",
            "Vorticity",
            "Vorticity2d",
            unit="s-1",
            preproc_func=vorticity_calculator.solve,
        )

        # Set initial conditions for current mesh iteration, ensuring exports are not overwritten
        if i == 0:
            options.apply_initial_conditions(solver_obj)
        else:
            solver_obj.create_exporters()
            assert q is not None
            uv, elev = q.split()
            solver_obj.assign_initial_conditions(uv=uv, elev=elev)
            solver_obj.i_export = i_export
            solver_obj.next_export_t = i_export * export_time
            solver_obj.iteration = int(
                np.ceil(solver_obj.next_export_t / options.timestep)
            )
            solver_obj.simulation_time = solver_obj.iteration * options.timestep
            solver_obj.export_initial_state = False
            for f in options.fields_to_export:
                solver_obj.exporters["vtk"].exporters[f].set_next_export_ix(
                    i_export + 1
                )
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
            t = solver_obj.simulation_time
            wq = export_time
            if np.isclose(t, start_time):
                wq *= 0.5
            elif t > options.simulation_end_time - options.timestep - t_epsilon:
                wq *= 0.5
            metrics[i] += wq * isotropic_metric(
                vorticity_2d, target_space=metrics[i].function_space()
            )

        # Solve forward on current subinterval, extracting vorticity metrics
        solver_obj.iterate(
            update_forcings=options.update_forcings, export_func=export_func
        )
        q = solver_obj.fields.solution_2d.copy(deepcopy=True)

        # Save metric data to file during first fixed point iteration
        if not converged:
            fname = initial_metric if fp_iteration == 0 else stashed_metric
            with DumbCheckpoint(fname.format(i, args.level), mode=FILE_CREATE) as chk:
                chk.store(metrics[i], name="Metric")
            viewer = PETSc.Viewer().createHDF5(stashed_mesh.format(i, args.level), "w")
            viewer(meshes[i].topology_dm)
    assert np.isclose(options.simulation_end_time, end_time)

    # Escape if converged
    if converged:
        print_output(f"Termination due to {converged_reason}")
        break

    # Apply space-time normalisation
    space_time_normalise(metrics, end_time, timesteps, args.target, args.norm_order)
    enforce_element_constraints(metrics, args.h_min, args.h_max)

    # Adapt meshes, checking for convergence of element count
    num_cells = []
    elements_converged = True
    for i, (metric, num_cells_old_i) in enumerate(zip(metrics, num_cells_old)):
        meshes[i] = Mesh(adapt(meshes[i], metric).coordinates)
        num_cells_i = meshes[i].num_cells()
        num_cells.append(num_cells_i)
        if np.abs(num_cells_i - num_cells_old_i) > args.element_rtol * num_cells_old_i:
            elements_converged = False
    if elements_converged:
        print_output(f"Mesh element count converged to rtol {args.element_rtol:.2e}")
        converged = True
        converged_reason = "converged element counts"
