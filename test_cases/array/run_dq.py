from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
import itertools
from options import ArrayOptions


num_tidal_cycles = 0.125
num_meshes = 5
target = 40000.0
norm_order = 1
h_min = 0.1
h_max = 1000.0
maxiter = 0  # TODO
element_rtol = 0.01
qoi_rtol = 0.01

# Mesh independent setup
options = ArrayOptions(level=0)
end_time = num_tidal_cycles*options.tide_time
output_dir = os.path.join(options.output_directory, 'fixed_mesh', 'level0')
options.output_directory = create_directory(output_dir)
Ct = options.quadratic_drag_coefficient
ct = options.corrected_thrust_coefficient*Constant(pi/8)
dt = options.timestep
timesteps = [dt]*num_meshes
dt_per_export = int(options.simulation_export_time/dt)
solves_per_dt = 1

# Initial mesh sequence
meshes = [Mesh(options.mesh2d.coordinates) for i in range(num_meshes)]


def solver(ic, t_start, t_end, dt, J=0, qoi=None, recover_vorticity=False):
    """
    Solve forward over time window
    (`t_start`, `t_end`) in P1DG-P1DG space.
    """
    mesh = ic.function_space().mesh()
    options.create_tidal_farm(mesh=mesh)
    P1 = get_functionspace(mesh, "CG", 1)
    options.horizontal_viscosity = options.set_viscosity(P1)
    options.simulation_end_time = t_end
    i_export = int(np.round(t_start/options.simulation_export_time))

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options, mesh=mesh)
    options.apply_boundary_conditions(solver_obj)
    options.J = J
    options.no_exports = True

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
        _Ct = _Ct + ct*interpolate(Constant(1.0), P0, subset=subset)

    def update_forcings(t):
        options.update_forcings(t)
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t, turbine_drag=_Ct)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=options.export_func)
    return solver_obj.fields.solution_2d, options.J


def time_integrated_qoi(sol, t, turbine_drag=Ct):
    """
    Power output of the array at time `t`.

    Integration in time gives the energy output.
    """
    u, eta = sol.split()
    return turbine_drag*pow(dot(u, u), 1.5)*dx


def initial_condition(fs):
    """
    Near-zero initial velocity and an
    initial elevation which satisfies
    the boundary conditions.
    """
    q = Function(fs)
    u, eta = q.split()
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
for fp_iteration in range(maxiter+1):
    if fp_iteration == maxiter:
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
    J, solutions = solve_adjoint(
        solver, initial_condition, time_integrated_qoi, spaces, end_time, dt,
        timesteps_per_export=dt_per_export, solves_per_timestep=solves_per_dt,
    )  # TODO: Option to just checkpoint only - use it if converged

    # Check for QoI convergence
    if J_old is not None:
        if abs(J - J_old) < qoi_rtol*J_old:
            converged = True
            converged_reason = 'converged quantity of interest'

    # # Escape if converged  # TODO
    # if converged:
    #     print_output(f"Termination due to {converged_reason}")
    #     break

    # Create vtu output files
    outdir = create_directory(os.path.join('outputs', 'fixed_mesh', f'{num_meshes}mesh'))
    outfiles = {
        'forward': File(os.path.join(outdir, 'Forward2d.pvd')),
        'forward_old': File(os.path.join(outdir, 'ForwardOld2d.pvd')),
        'adjoint_next': File(os.path.join(outdir, 'AdjointNext2d.pvd')),
        'adjoint': File(os.path.join(outdir, 'Adjoint2d.pvd')),
    }
    fields = ['forward', 'forward_old', 'adjoint_next', 'adjoint']

    # Loop over all meshes to evaluate error indicators
    difference_quotients = []
    for i, mesh in enumerate(meshes):
        for f in fields:
            outfiles[f]._topology = None  # Allow writing a different mesh

        # Update FarmOptions object according to mesh
        options.create_tidal_farm(mesh=mesh)
        P1 = get_functionspace(mesh, "CG", 1)
        options.horizontal_viscosity = options.set_viscosity(P1)

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
            _dq = ee.difference_quotient(*args)

            # Apply trapezium rule
            if j in (0, N-1):
                _dq *= 0.5
            _dq *= dt
            dq += _dq
        difference_quotients.append(dq)

    # Plot difference quotient
    outfiles['error'] = File(os.path.join(outdir, 'Indicator2d.pvd'))
    for dq in reversed(difference_quotients):
        outfiles['error']._topology = None
        outfiles['error'].write(dq)

    # Construct isotropic metrics
    metrics = [
        isotropic_metric(dq)
        for dq in difference_quotients
    ]
    space_time_normalise(metrics, end_time, timesteps, target, norm_order)
    enforce_element_constraints(metrics, h_min, h_max)

    # Plot metrics
    outfiles['metric'] = File(os.path.join(outdir, 'Metric2d.pvd'))
    for metric in reversed(metrics):
        metric.rename("Metric")
        outfiles['metric']._topology = None
        outfiles['metric'].write(metric)

    # Adapt meshes
    for i, metric in enumerate(metrics):
        meshes[i] = Mesh(adapt(meshes[i], metric).coordinates)
    num_cells = [mesh.num_cells() for mesh in meshes]

    # Plot meshes
    outfiles['mesh'] = File(os.path.join(outdir, 'Mesh2d.pvd'))
    for mesh in reversed(meshes):
        outfiles['mesh']._topology = None
        outfiles['mesh'].write(mesh.coordinates)

    # Check for convergence of element count
    elements_converged = False
    if num_cells_old is not None:
        elements_converged = True
        for nc, _nc in zip(num_cells, num_cells_old):
            if abs(nc - _nc) > element_rtol*_nc:
                elements_converged = False
    num_cells_old = num_cells
    if elements_converged:
        print_output(f"Mesh element count converged to rtol {element_rtol}")
        converged = True
        converged_reason = 'converged element counts'
