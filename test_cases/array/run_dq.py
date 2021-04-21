from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
import itertools
from options import ArrayOptions


end_time = 8928.0
num_meshes = 80

# Set parameters
options = ArrayOptions(level=0)
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh', 'level0')
options.output_directory = create_directory(output_dir)

# Create MixedFunctionSpace
U_2d = VectorFunctionSpace(options.mesh2d, "DG", 1, name="U_2d")
H_2d = get_functionspace(options.mesh2d, "DG", 1, name="H_2d")
function_space = MixedFunctionSpace([U_2d, H_2d])

# Turbine parametrisation
Ct = options.quadratic_drag_coefficient
ct = options.corrected_thrust_coefficient*Constant(pi/8)
P0 = FunctionSpace(options.mesh2d, "DG", 0)
for i, subdomain_id in enumerate(options.farm_ids):
    subset = options.mesh2d.cell_subset(subdomain_id)
    Ct = Ct + ct*interpolate(Constant(1.0), P0, subset=subset)


def solver(ic, t_start, t_end, dt, J=0, qoi=None, recover_vorticity=False):
    """
    Solve forward over time window
    (`t_start`, `t_end`) in P1DG-P1DG space.
    """
    options.simulation_end_time = t_end
    # if np.isclose(t_end, end_time):
    #     options.simulation_end_time += 0.5*options.timestep
    i_export = int(t_start/options.simulation_export_time)

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
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

    def update_forcings(t):
        options.update_forcings(t)
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=options.export_func)
    return solver_obj.fields.solution_2d, options.J


def time_integrated_qoi(sol, t):
    """
    Power output of the array at time `t`.

    Integration in time gives the energy output.
    """
    u, eta = sol.split()
    return Ct*pow(dot(u, u), 1.5)*dx


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


# Setup
J = 0
fs = function_space
dt = options.timestep
dt_per_export = int(options.simulation_export_time/dt)
solves_per_dt = 1
qoi = time_integrated_qoi
num_timesteps = int(end_time/dt)
spaces = [fs]*num_meshes

# Solve forward and adjoint on each subinterval
J, solutions = solve_adjoint(
    solver, initial_condition, qoi, spaces, end_time, dt,
    timesteps_per_export=dt_per_export, solves_per_timestep=solves_per_dt,
)

# Plot solutions
outdir = create_directory(os.path.join('outputs', 'fixed_mesh', f'{num_meshes}mesh'))
fwd_outfile = File(os.path.join(outdir, 'Forward2d.pvd'))
fwd_old_outfile = File(os.path.join(outdir, 'ForwardOld2d.pvd'))
adj_outfile = File(os.path.join(outdir, 'Adjoint2d.pvd'))
adj_next_outfile = File(os.path.join(outdir, 'AdjointNext2d.pvd'))
err_outfile = File(os.path.join(outdir, 'Indicator2d.pvd'))
difference_quotients = []
for i in range(num_meshes):

    # Create error estimator
    ee = ErrorEstimator(options, mesh=options.mesh2d)  # NOTE: mesh will be passed in here
    dq = Function(ee.P0, name="Difference quotient")

    N = len(solutions['adjoint'][i])
    for j in range(N):
        if i < num_meshes-1 and j == N-1:
            continue

        # Forward
        u, eta = solutions['forward'][i][j].split()
        u.rename("Velocity")
        eta.rename("Elevation")
        fwd_outfile.write(u, eta)

        # Forward old
        u_old, eta_old = solutions['forward_old'][i][j].split()
        u_old.rename("Velocity")
        eta_old.rename("Elevation")
        fwd_old_outfile.write(u_old, eta_old)

        # Adjoint
        z, zeta = solutions['adjoint'][i][j].split()
        z.rename("Adjoint velocity")
        zeta.rename("Adjoint elevation")
        adj_outfile.write(z, zeta)

        # Adjoint next
        z_next, zeta_next = solutions['adjoint_next'][i][j].split()
        z_next.rename("Adjoint velocity")
        zeta_next.rename("Adjoint elevation")
        adj_next_outfile.write(z_next, zeta_next)

        # Error indicator
        _dq = ee.difference_quotient(u, eta, u_old, eta_old, z_next, zeta_next, z, zeta)

        # Apply trapezium rule
        if j in (0, N-1):
            _dq *= 0.5
        _dq *= dt
        dq += _dq
    difference_quotients.append(dq)
for dq in reversed(difference_quotients):
    err_outfile.write(dq)
