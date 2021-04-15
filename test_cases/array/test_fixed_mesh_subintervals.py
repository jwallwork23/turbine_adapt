from turbine_adapt import *
from firedrake.adjoint.blocks import GenericSolveBlock
import pyadjoint
import itertools
from options import ArrayOptions


end_time = 892.8

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
    if np.isclose(t_end, end_time):
        options.simulation_end_time += 0.5*options.timestep
    i_export = int(t_start/options.simulation_export_time)

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
    options.apply_boundary_conditions(solver_obj)
    options.J = J

    # Callback which recovers vorticity
    if recover_vorticity:
        cb = PeakVorticityCallback(solver_obj)
        cb._create_new_file = i == 0
        solver_obj.add_callback(cb, 'timestep')
        if i > 0:
            cb._outfile.counter = itertools.count(start=i_export + 1)  # FIXME

    # Set initial conditions for current mesh iteration
    if i == 0:
        options.apply_initial_conditions(solver_obj)
    else:
        solver_obj.create_exporters()
        uv, elev = ic.split()
        solver_obj.assign_initial_conditions(uv=uv, elev=elev)
        solver_obj.i_export = i_export
        solver_obj.next_export_t = i_export*options.simulation_export_time
        solver_obj.iteration = int(np.ceil(solver_obj.next_export_t/options.timestep))
        solver_obj.simulation_time = t_start
        solver_obj.export_initial_state = False
        for f in options.fields_to_export:
            solver_obj.exporters['vtk'].exporters[f].set_next_export_ix(i_export + 1)

    def update_forcings(t):
        options.update_forcings(t)
        if qoi is not None:
            options.J += qoi(solver_obj.fields.solution_2d, t)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=options.export_func)
    return solver_obj.fields.solution_2d.copy(deepcopy=True), options.J


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


# ---------------------------
# standard tests for pytest
# ---------------------------

def test_adjoint_same_mesh():
    """
    **Disclaimer: largely copied from pyroteus/test/test_adjoint.py
    """
    from firedrake_adjoint import Control

    # Setup
    print_output("\n--- Setting up\n")
    J = 0
    fs = function_space
    dt = options.timestep
    dt_per_export = int(options.simulation_export_time/dt)
    solves_per_dt = 1
    qoi = time_integrated_qoi
    num_timesteps = int(end_time/dt)

    # Solve forward and adjoint without subinterval framework
    print_output("\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    solver_kwargs = dict(qoi=lambda *args: assemble(qoi(*args)))
    ic = initial_condition(fs)
    sol, J = solver(ic, 0.0, end_time, dt, **solver_kwargs)
    pyadjoint.compute_gradient(J, Control(ic))  # FIXME: gradient w.r.t. mixed function not correct
    tape = pyadjoint.get_working_tape()
    solve_blocks = [
        block for block in tape.get_blocks()
        if issubclass(block.__class__, GenericSolveBlock)
        and block.adj_sol is not None
    ][-num_timesteps*solves_per_dt::solves_per_dt]
    final_adj_sols = [solve_blocks[0].adj_sol]
    qois = [J]

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        N = len(spaces)
        plural = '' if N == 1 else 's'
        print_output(f"\n --- Solving the adjoint problem on {N} subinterval{plural} using pyroteus\n")
        # Solve forward and adjoint on each subinterval
        J, adj_sols = solve_adjoint(
            solver, initial_condition, qoi, spaces, end_time, dt,
            timesteps_per_export=dt_per_export, solves_per_timestep=solves_per_dt,
        )
        final_adj_sols.append(adj_sols[-1][-1])
        qois.append(J)

        # Check energy outputs match
        assert np.isclose(*qois[::N]), f"QoIs do not match ({qois[0]} vs. {qois[-1]})"

        # Check adjoint solutions at initial time match
        err = errornorm(*final_adj_sols[::N])/norm(final_adj_sols[0])
        assert np.isclose(err, 0.0), f"Non-zero adjoint error ({err})"


# ---------------------------
# debugging
# ---------------------------

if __name__ == "__main__":
    """
    Solve over the subintervals in sequence
    """
    num_meshes = 5
    q = initial_condition(function_space)
    J = 0
    dt = options.timestep
    mesh_iteration_time = end_time/num_meshes
    qoi = time_integrated_qoi
    for i in range(num_meshes):
        t_start = i*mesh_iteration_time
        t_end = (i+1)*mesh_iteration_time
        q, J = solver(q, t_start, t_end, dt, J=J, qoi=lambda *args: assemble(qoi(*args)))
    print_output(f"Energy output = {J/3600000} kW h")
