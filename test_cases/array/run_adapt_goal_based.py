from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from pyadjoint import stop_annotating
import itertools
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/run_goal_based.py')
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
parser.add_argument('-target', 5000.0)
parser.add_argument('-h_min', 0.01)
parser.add_argument('-h_max', 100.0)
parser.add_argument('-turbine_h_max', 10.0)
parser.add_argument('-adjoint_projection', True)
parser.add_argument('-flux_form', False)
parser.add_argument('-space_only', False)
parser.add_argument('-approach', 'isotropic_dwr')
parser.add_argument('-error_indicator', 'difference_quotient')
parser.add_argument('-load_index', 0)
parser.add_argument('-staggered', False, help="""
    Toggle between aligned and staggered array
    (default False).""")
parsed_args = parser.parse_args()
num_meshes = parsed_args.num_meshes
if parsed_args.approach not in ('isotropic_dwr', 'anisotropic_dwr'):  # NOTE: anisotropic => weighted Hessian
    raise ValueError(f"Adaptation approach {parsed_args.approach} not recognised.")
if parsed_args.error_indicator != 'difference_quotient':
    raise NotImplementedError(f"Error indicator {parsed_args.error_indicator} not recognised.")
approach = parsed_args.approach.split('_')[0]
hmax = Constant(parsed_args.h_max)
turbine_hmax = Constant(parsed_args.turbine_h_max)

# Mesh independent setup
ramp_dir = os.path.join('outputs', 'fixed_mesh', f'level{parsed_args.ramp_level}')
options = ArrayOptions(
    level=parsed_args.level, staggered=parsed_args.staggered,
    ramp_level=parsed_args.ramp_level, ramp_dir=ramp_dir,
)
end_time = parsed_args.num_tidal_cycles*options.tide_time
root_dir = os.path.join(options.output_directory, approach)
output_dir = os.path.join(root_dir, f'target{parsed_args.target:.0f}')
options.output_directory = create_directory(output_dir)
Ct = options.quadratic_drag_coefficient
ct = options.corrected_thrust_coefficient*Constant(pi/8)
dt = options.timestep
target = parsed_args.target
if not parsed_args.space_only:
    target *= end_time/dt  # space-time complexity
timesteps = [dt]*num_meshes
dt_per_export = int(options.simulation_export_time/dt)
solves_per_dt = 1

# Initial mesh sequence
meshes = [Mesh(options.mesh2d.coordinates) for i in range(num_meshes)]


def solver(ic, t_start, t_end, dt, J=0, qoi=None, **model_options):
    """
    Solve forward over time window
    (`t_start`, `t_end`) in P1DG-P1DG space.
    """
    mesh = ic.function_space().mesh()
    options.rebuild_mesh_dependent_components(mesh)
    options.simulation_end_time = t_end
    i_export = int(np.round(t_start/options.simulation_export_time))

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options, mesh=mesh)
    options.apply_boundary_conditions(solver_obj)
    options.J = J
    recover_vorticity = model_options.pop('recover_vorticity', False)
    compute_power = model_options.pop('compute_power', False)
    model_options.setdefault('no_exports', True)
    options.update(model_options)
    if not options.no_exports:
        options.fields_to_export = ['uv_2d', 'elev_2d']

    # Callback which recovers vorticity
    if recover_vorticity:
        cb = PeakVorticityCallback(solver_obj)
        cb._create_new_file = i_export == 0
        solver_obj.add_callback(cb, 'timestep')
        if i_export > 0:
            cb._outfile.counter = itertools.count(start=i_export + 1)  # FIXME

    # Callback which writes power output to HDF5
    if compute_power:
        cb = PowerOutputCallback(solver_obj)
        cb._create_new_file = i_export == 0
        solver_obj.add_callback(cb, 'timestep')

    # Set initial conditions for current mesh iteration
    solver_obj.create_exporters()
    uv, elev = ic.split()
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)
    solver_obj.i_export = i_export
    solver_obj.next_export_t = i_export*options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/options.timestep))
    solver_obj.simulation_time = t_start
    solver_obj.export_initial_state = False
    if not options.no_exports:
        solver_obj.exporters['vtk'].set_next_export_ix(i_export)

    # Turbine parametrisation
    P0 = FunctionSpace(mesh, "DG", 0)
    _Ct = Constant(Ct)
    for i, subdomain_id in enumerate(options.farm_ids):  # TODO: Use union
        subset = mesh.cell_subset(subdomain_id)
        _Ct = _Ct + interpolate(ct, P0, subset=subset)

    def update_forcings(t):
        options.update_forcings(t)
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t, turbine_drag=_Ct)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=options.export_func)
    return solver_obj.fields.solution_2d, options.J


def time_integrated_qoi(sol, t, turbine_drag=None):
    """
    Power output of the array at time `t`.

    Integration in time gives the energy output.
    """
    assert turbine_drag is not None, \
        "Turbine drag needs to be provided."
    u, eta = sol.split()
    unorm = sqrt(dot(u, u))
    return turbine_drag*pow(unorm, 3)*dx


def initial_condition(fs):
    """
    Near-zero initial velocity and an
    initial elevation which satisfies
    the boundary conditions.
    """
    q = Function(fs)
    u, eta = q.split()
    if options.ramp is not None:
        print_output("Initialising with ramped hydrodynamics")
        u_ramp, eta_ramp = options.ramp.split()
        u.project(u_ramp)
        eta.project(eta_ramp)
    else:
        print_output("Initialising with unramped hydrodynamics")
        u.interpolate(as_vector([1e-8, 0.0]))
        x, y = SpatialCoordinate(fs.mesh())
        X = 2*x/options.domain_length  # Non-dimensionalised x
        eta.interpolate(-options.max_amplitude*X)
    return q


# Enter fixed point iteration
miniter = parsed_args.miniter
maxiter = parsed_args.maxiter
if miniter > maxiter:
    miniter = maxiter
converged = False
converged_reason = None
num_cells_old = None
J_old = None
load_index = parsed_args.load_index
fp_iteration = load_index
while fp_iteration <= maxiter:
    outfiles = {}
    if fp_iteration < miniter:
        converged = False
    elif fp_iteration == maxiter:
        converged = True
        if converged_reason is None:
            converged_reason = 'maximum number of iterations reached'

    # Load meshes, if requested
    if load_index > 0 and fp_iteration == load_index:
        for i in range(num_meshes):
            mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}")
            if not os.path.exists(mesh_fname + '.h5'):
                raise IOError(f"Cannot load mesh file {mesh_fname}.")
            plex = PETSc.DMPlex().create()
            plex.createFromFile(mesh_fname + '.h5')
            meshes[i] = Mesh(plex)

    # Create function spaces
    spaces = [
        MixedFunctionSpace([
            VectorFunctionSpace(mesh, "DG", 1, name="U_2d"),
            get_functionspace(mesh, "DG", 1, name="H_2d"),
        ])
        for mesh in meshes
    ]
    metrics = [
        Function(TensorFunctionSpace(mesh, "CG", 1), name="Metric")
        for mesh in meshes
    ]

    # Load metric data for first iteration if available
    loaded = False
    if fp_iteration == load_index:
        for i, metric in enumerate(metrics):
            if load_index == 0:
                metric_fname = os.path.join(root_dir, f'metric{i}')
            else:
                metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
            if os.path.exists(metric_fname + '.h5'):
                print_output(f"\n--- Loading metric data on mesh {i+1}\n")
                loaded = True
                with DumbCheckpoint(metric_fname, mode=FILE_READ) as chk:
                    chk.load(metric, name="Metric")
            else:
                assert not loaded, "Only partial metric data available"
                break

    # Otherwise, solve forward and adjoint
    if not loaded:

        # Solve forward and adjoint on each subinterval
        args = (solver, initial_condition, time_integrated_qoi, spaces, end_time, dt)
        if converged:
            with stop_annotating():
                print_output("\n--- Final forward run\n")
                J, checkpoints = get_checkpoints(
                    *args, timesteps_per_export=dt_per_export,
                    solver_kwargs=dict(no_exports=False, compute_power=True),
                )
        else:
            print_output(f"\n--- Forward-adjoint sweep {fp_iteration}\n")
            J, solutions = solve_adjoint(
                *args, timesteps_per_export=dt_per_export,
                solves_per_timestep=solves_per_dt,
                adjoint_projection=parsed_args.adjoint_projection,
            )

        # Check for QoI convergence
        if J_old is not None:
            if abs(J - J_old) < parsed_args.qoi_rtol*J_old and fp_iteration >= miniter:
                converged = True
                converged_reason = 'converged quantity of interest'
                with stop_annotating():
                    print_output("\n--- Final forward run\n")
                    J, checkpoints = get_checkpoints(
                        *args, timesteps_per_export=dt_per_export,
                        solver_kwargs=dict(no_exports=False, compute_power=True),
                    )
        J_old = J

        # Escape if converged
        if converged:
            print_output(f"Termination due to {converged_reason} after {fp_iteration+1} iterations")
            print_output(f"Energy output: {J/3.6e+09} MWh")
            break

        # Create vtu output files
        outfiles['forward'] = File(os.path.join(output_dir, 'Forward2d.pvd'))
        outfiles['forward_old'] = File(os.path.join(output_dir, 'ForwardOld2d.pvd'))
        outfiles['adjoint_next'] = File(os.path.join(output_dir, 'AdjointNext2d.pvd'))
        outfiles['adjoint'] = File(os.path.join(output_dir, 'Adjoint2d.pvd'))
        fields = ['forward', 'forward_old', 'adjoint_next', 'adjoint']

        # Construct metric
        error_indicators = []
        hessians = []
        with stop_annotating():
            print_output(f"\n--- Error estimation {fp_iteration}\n")
            for i, mesh in enumerate(meshes):
                for f in fields:
                    outfiles[f]._topology = None  # Allow writing a different mesh
                options.rebuild_mesh_dependent_components(mesh)
                options.get_bnd_conditions(spaces[i])
                update_forcings = options.update_forcings

                # Create error estimator
                ee = ErrorEstimator(options, mesh=mesh, error_estimator=parsed_args.error_indicator)
                if approach == 'isotropic':
                    error_indicators_step = [Function(ee.P0, name="Error indicator")]
                else:
                    error_indicators_step = [Function(ee.P0) for field in range(3)]
                hessians_step = [] if approach == 'isotropic' else [Function(metrics[i]) for field in range(3)]

                # Loop over all exported timesteps
                N = len(solutions['adjoint'][i])
                for j in range(N):
                    if i < num_meshes-1 and j == N-1:
                        continue

                    # Plot fields
                    args = []
                    for f in fields:
                        args.extend(solutions[f][i][j].split())
                        args[-2].rename("Adjoint velocity" if 'adjoint' in f else "Velocity")
                        args[-1].rename("Adjoint elevation" if 'adjoint' in f else "Elevation")
                        outfiles[f].write(*args[-2:])

                    # Evaluate error indicator
                    update_forcings(i*end_time/num_meshes + dt*(j + 1))
                    if approach == 'isotropic':
                        _error_indicators_step = [ee.error_indicator(*args, flux_form=parsed_args.flux_form)]
                        _hessians_step = []
                    else:
                        _error_indicators_step = ee.strong_residuals(*args[:4])
                        _hessians_step = ee.recover_hessians(*args[6:])
                        for _hessian_next, _hessian in zip(ee.recover_hessians(*args[4:6]), _hessians_step):
                            _hessian += _hessian_next
                            _hessian *= 0.5

                    # Apply trapezium rule
                    if j in (0, N-1):
                        for _error_indicator in _error_indicators_step:
                            _error_indicator *= 0.5
                        for _H_i in _hessians_step:
                            _H_i *= 0.5
                    for error_indicator, _error_indicator in zip(error_indicators_step, _error_indicators_step):
                        _error_indicator *= dt
                        error_indicator += _error_indicator
                    for hessian, _hessian in zip(hessians_step, _hessians_step):
                        _hessian *= dt
                        hessian += _hessian
                error_indicators.append(error_indicators_step)
                hessians.append(hessians_step)

            # Plot error indicators
            if approach == 'isotropic':
                outfiles['error'] = File(os.path.join(output_dir, 'Indicator2d.pvd'))
                for error_indicator in error_indicators:
                    outfiles['error']._topology = None
                    outfiles['error'].write(error_indicator[0])

            # Construct metrics
            for i, error_indicator in enumerate(error_indicators):
                if approach == 'isotropic':
                    metrics[i].assign(isotropic_metric(error_indicator[0]))
                else:
                    metrics[i].assign(anisotropic_metric(error_indicator, hessians[i], element_wise=False))

                print_output(f"\n--- Storing metric data on mesh {i+1}\n")
                metric_fname = os.path.join(output_dir, f'metric{i}_fp{fp_iteration}')
                with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                    chk.store(metrics[i], name="Metric")
                if fp_iteration == 0:
                    metric_fname = os.path.join(root_dir, f'metric{i}')
                    with DumbCheckpoint(metric_fname, mode=FILE_CREATE) as chk:
                        chk.store(metrics[i], name="Metric")

    # Process metrics
    print_output(f"\n--- Metric processing {fp_iteration}\n")
    if parsed_args.space_only:
        for metric in metrics:
            space_normalise(metric, target, parsed_args.norm_order)
    else:
        metrics = space_time_normalise(
            metrics, end_time, timesteps, target, parsed_args.norm_order
        )

    # Enforce element constraints, accounting for turbines
    h_max = []
    for mesh in meshes:
        expr = Constant(hmax)
        P0 = FunctionSpace(mesh, "DG", 0)
        for i, subdomain_id in enumerate(options.farm_ids):  # TODO: Use union
            subset = mesh.cell_subset(subdomain_id)
            expr = expr + interpolate(turbine_hmax - hmax, P0, subset=subset)
        hmax_func = interpolate(expr, FunctionSpace(mesh, "CG", 1))
        h_max.append(hmax_func)
    metrics = enforce_element_constraints(
        metrics, parsed_args.h_min, h_max
    )

    # Plot metrics
    outfiles['metric'] = File(os.path.join(output_dir, 'Metric2d.pvd'))
    for metric in metrics:
        metric.rename("Metric")
        outfiles['metric']._topology = None
        outfiles['metric'].write(metric)

    # Adapt meshes
    print_output(f"\n--- Mesh adaptation {fp_iteration}\n")
    outfiles['mesh'] = File(os.path.join(output_dir, 'Mesh2d.pvd'))
    for i, metric in enumerate(metrics):
        meshes[i] = Mesh(adapt(meshes[i], metric).coordinates)
        outfiles['mesh']._topology = None
        outfiles['mesh'].write(meshes[i].coordinates)
    num_cells = [mesh.num_cells() for mesh in meshes]

    # Check for convergence of element count
    elements_converged = False
    if num_cells_old is not None:
        elements_converged = True
        for nc, _nc in zip(num_cells, num_cells_old):
            if abs(nc - _nc) > parsed_args.element_rtol*_nc:
                elements_converged = False
    num_cells_old = num_cells
    if elements_converged:
        print_output(f"Mesh element count converged to rtol {parsed_args.element_rtol}")
        converged = True
        converged_reason = 'converged element counts'

    # Save mesh data to disk
    if COMM_WORLD.size == 1:
        for i, mesh in enumerate(meshes):
            mesh_fname = os.path.join(output_dir, f"mesh_fp{fp_iteration}_{i}")
            viewer = PETSc.Viewer().createHDF5(mesh_fname, 'w')
            viewer(mesh.topology_dm)

    # Increment
    fp_iteration += 1

# Log convergence reason
with open(os.path.join(output_dir, 'log'), 'a+') as f:
    f.write(f"Converged in {fp_iteration+1} iterations due to {converged_reason}")
