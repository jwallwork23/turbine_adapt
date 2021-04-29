from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from pyadjoint import stop_annotating
import itertools
from options import ArrayOptions


# Parse arguments
parser = Parser(prog='turbine/array/run_goal_based.py')
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-num_tidal_cycles', 1.0)
parser.add_argument('-num_meshes', 80)
parser.add_argument('-miniter', 3)
parser.add_argument('-maxiter', 5)
parser.add_argument('-element_rtol', 0.005)
parser.add_argument('-qoi_rtol', 0.005)
parser.add_argument('-norm_order', 1.0)
parser.add_argument('-target', 5000.0)
parser.add_argument('-h_min', 0.01)
parser.add_argument('-h_max', 100.0)
# parser.add_argument('-adjoint_projection', True)  # FIXME
parser.add_argument('-adjoint_projection', False)
parser.add_argument('-flux_form', False)
parser.add_argument('-space_only', False)
parsed_args = parser.parse_args()
num_meshes = parsed_args.num_meshes

# Mesh independent setup
options = ArrayOptions(ramp_dir=os.path.join('outputs', 'fixed_mesh', f'level{parsed_args.level}'))
end_time = parsed_args.num_tidal_cycles*options.tide_time
output_dir = os.path.join(options.output_directory, 'dwr', f'target{parsed_args.target:.0f}')
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

    # Set initial conditions for current mesh iteration
    solver_obj.create_exporters()
    uv, elev = ic.split()
    solver_obj.assign_initial_conditions(uv=uv, elev=elev)
    solver_obj.i_export = i_export
    solver_obj.next_export_t = i_export*options.simulation_export_time
    solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/options.timestep))
    solver_obj.simulation_time = t_start
    solver_obj.export_initial_state = False

    # Turbine parametrisation
    P0 = FunctionSpace(mesh, "DG", 0)
    _Ct = Constant(Ct)
    for i, subdomain_id in enumerate(options.farm_ids):
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
converged = False
converged_reason = None
num_cells_old = None
J_old = None
for fp_iteration in range(parsed_args.maxiter+1):
    if fp_iteration < parsed_args.miniter:
        converged = False
    elif fp_iteration == parsed_args.maxiter:
        converged = True
        if converged_reason is None:
            converged_reason = 'maximum number of iterations reached'

    # Create function spaces
    spaces = [
        MixedFunctionSpace([
            VectorFunctionSpace(mesh, "DG", 1, name="U_2d"),
            get_functionspace(mesh, "DG", 1, name="H_2d"),
        ])
        for mesh in meshes
    ]

    # Solve forward and adjoint on each subinterval
    args = (solver, initial_condition, time_integrated_qoi, spaces, end_time, dt)
    if converged:
        with stop_annotating():
            print_output("\n--- Final forward run\n")
            J, checkpoints = get_checkpoints(
                *args, timesteps_per_export=dt_per_export, solver_kwargs=dict(no_exports=False),
            )
    else:
        print_output(f"\n--- Forward-adjoint sweep {fp_iteration+1}\n")
        J, solutions = solve_adjoint(
            *args, timesteps_per_export=dt_per_export,
            solves_per_timestep=solves_per_dt,
            adjoint_projection=parsed_args.adjoint_projection,
        )

    # Check for QoI convergence
    if J_old is not None:
        if abs(J - J_old) < parsed_args.qoi_rtol*J_old and fp_iteration < parsed_args.miniter:
            converged = True
            converged_reason = 'converged quantity of interest'
            with stop_annotating():
                print_output("\n--- Final forward run\n")
                J, checkpoints = get_checkpoints(
                    *args, timesteps_per_export=dt_per_export, solver_kwargs=dict(no_exports=False),
                )
    J_old = J

    # Escape if converged
    if converged:
        print_output(f"Termination due to {converged_reason} after {fp_iteration+1} iterations")
        print_output(f"Energy output: {J/3.6e+09} MWh")
        break

    # Create vtu output files
    outfiles = {
        'forward': File(os.path.join(output_dir, 'Forward2d.pvd')),
        'forward_old': File(os.path.join(output_dir, 'ForwardOld2d.pvd')),
        'adjoint_next': File(os.path.join(output_dir, 'AdjointNext2d.pvd')),
        'adjoint': File(os.path.join(output_dir, 'Adjoint2d.pvd')),
    }
    fields = ['forward', 'forward_old', 'adjoint_next', 'adjoint']

    # Loop over all meshes to evaluate error indicators
    difference_quotients = []
    with stop_annotating():
        print_output(f"\n--- Error estimation {fp_iteration+1}\n")
        for i, mesh in enumerate(meshes):
            for f in fields:
                outfiles[f]._topology = None  # Allow writing a different mesh
            options.rebuild_mesh_dependent_components(mesh)
            options.get_bnd_conditions(spaces[i])
            update_forcings = options.update_forcings

            # Create error estimator
            ee = ErrorEstimator(options, mesh=mesh)
            dq = Function(ee.P0, name="Difference quotient")

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
                _dq = ee.difference_quotient(*args, flux_form=parsed_args.flux_form)

                # Apply trapezium rule
                if j in (0, N-1):
                    _dq *= 0.5
                _dq *= dt
                dq += _dq
            difference_quotients.append(dq)

        # Plot difference quotient
        outfiles['error'] = File(os.path.join(output_dir, 'Indicator2d.pvd'))
        for dq in difference_quotients:
            outfiles['error']._topology = None
            outfiles['error'].write(dq)

        # Construct isotropic metrics
        metrics = [
            isotropic_metric(dq)
            for dq in difference_quotients
        ]
        if parsed_args.space_only:
            for metric in metrics:
                space_normalise(metric, target, parsed_args.norm_order)
        else:
            metrics = space_time_normalise(
                metrics, end_time, timesteps, target, parsed_args.norm_order
            )
        metrics = enforce_element_constraints(
            metrics, parsed_args.h_min, parsed_args.h_max
        )

        # Plot metrics
        outfiles['metric'] = File(os.path.join(output_dir, 'Metric2d.pvd'))
        for metric in metrics:
            metric.rename("Metric")
            outfiles['metric']._topology = None
            outfiles['metric'].write(metric)

        # Adapt meshes
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
