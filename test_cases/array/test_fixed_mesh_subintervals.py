from turbine_adapt import *
from firedrake_adjoint import Control
import pyadjoint
import itertools
from options import ArrayOptions


end_time = 89.28

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


def solver(ic, t_start, t_end, dt, J=0, qoi=None, **model_options):
    """
    Solve forward over time window
    (`t_start`, `t_end`) in P1DG-P1DG space.
    """
    recover_vorticity = model_options.pop('recover_vorticity', False)
    model_options.setdefault('no_exports', True)
    options.simulation_end_time = t_end
    # if np.isclose(t_end, end_time):
    #     options.simulation_end_time += 0.5*options.timestep
    i_export = int(t_start/options.simulation_export_time)

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
    options.apply_boundary_conditions(solver_obj)
    options.J = J
    options.update(model_options)
    options.fields_to_export = [] if options.no_exports else ['uv_2d', 'elev_2d']

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
    for f in options.fields_to_export:
        solver_obj.exporters['vtk'].exporters[f].set_next_export_ix(i_export + 1)

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


# ---------------------------
# standard tests for pytest
# ---------------------------

def test_adjoint_same_mesh(plot=False):
    """
    **Disclaimer: largely copied from
        pyroteus/test/test_adjoint.py
    """

    # Setup
    print_output("\n--- Setting up\n")
    fs = function_space
    dt = options.timestep
    dt_per_export = int(options.simulation_export_time/dt)
    qoi = time_integrated_qoi
    time_partition = TimePartition(
        end_time, 1, dt, timesteps_per_export=dt_per_export, solves_per_timestep=1,
    )

    # Solve forward and adjoint without subinterval framework
    print_output("\n--- Solving the adjoint problem on 1 subinterval using pyadjoint\n")
    solver_kwargs = dict(qoi=lambda *args: assemble(qoi(*args)))
    ic = initial_condition(fs)
    sol, J = solver(ic, 0.0, end_time, dt, **solver_kwargs)
    pyadjoint.compute_gradient(J, Control(ic))  # FIXME: gradient w.r.t. mixed function not correct
    _adj_sol = time_partition.solve_blocks()[0].adj_sol.copy(deepcopy=True)
    _J = J

    # Loop over having one or two subintervals
    for spaces in ([fs], [fs, fs]):
        N = len(spaces)
        print_output(f"\n --- Solving the adjoint problem on {N} subinterval"
                     + f"{'' if N == 1 else 's'} using pyroteus\n")

        # Solve forward and adjoint on each subinterval
        time_partition = TimePartition(end_time, N, dt, timesteps_per_export=dt_per_export)
        J, solutions = solve_adjoint(solver, initial_condition, qoi, spaces, time_partition)

        # Plot adjoint solutions
        if plot:
            outfile = File(f'outputs/fixed_mesh/Adjoint2d_{N}mesh/Adjoint2d.pvd')
            for i in reversed(range(N)):
                for j, adj_sol in enumerate(reversed(solutions['adjoint'][i])):
                    if i < N-1 and j == len(solutions['adjoint'][i])-1:
                        continue
                    z, zeta = adj_sol.split()
                    z.rename("Adjoint velocity")
                    zeta.rename("Adjoint elevation")
                    outfile.write(z, zeta)

        # Check energy outputs match
        assert np.isclose(_J, J), f"{N} meshes: QoIs do not match ({_J} vs. {J})"

        # Check adjoint solutions at initial time match
        err = errornorm(_adj_sol, solutions['adjoint'][0][0])/norm(_adj_sol)
        assert np.isclose(err, 0.0), f"{N} meshes: Non-zero adjoint error ({err})"


# ---------------------------
# debugging
# ---------------------------

if __name__ == "__main__":
    """
    Solve over the subintervals in sequence
    """
    # num_subintervals = 5
    num_subintervals = 1
    q = initial_condition(function_space)
    control = Control(q)  # FIXME: gradient w.r.t. mixed function not correct
    J = 0
    dt = options.timestep
    time_partition = TimePartition(
        end_time, num_subintervals, dt,
        timesteps_per_export=int(options.simulation_export_time/dt),
    )
    for i in range(num_subintervals):
        q, J = solver(
            q, *time_partition[i], J=J, no_exports=False,
            qoi=lambda *args: assemble(time_integrated_qoi(*args)),
        )
    print_output(f"Energy output = {J/3600000} kW h")
    pyadjoint.compute_gradient(J, control)
    solve_blocks = time_partition.solve_blocks()
    outfile = File('outputs/fixed_mesh/Adjoint2d/Adjoint2d.pvd')
    z, zeta = Function(solve_blocks[0].function_space).split()
    z.rename("Adjoint velocity")
    zeta.rename("Adjoint elevation")
    outfile.write(z, zeta)
    for j in reversed(range(0, len(solve_blocks), dt_per_export)):
        z, zeta = solve_blocks[j].adj_sol.split()
        z.rename("Adjoint velocity")
        zeta.rename("Adjoint elevation")
        outfile.write(z, zeta)
