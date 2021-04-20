from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from firedrake_adjoint import Control
from firedrake.adjoint.blocks import GenericSolveBlock
import pyadjoint
import itertools
from options import ArrayOptions
import os


def index_str(index, n=5):
    return (n - len(str(index)))*'0' + str(index)


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
    # if np.isclose(t_end, end_time):
    #     options.simulation_end_time += 0.5*options.timestep
    i_export = int(t_start/options.simulation_export_time)

    # Create a new solver object and assign boundary conditions
    solver_obj = FarmSolver(options)
    options.apply_boundary_conditions(solver_obj)
    options.J = J
    options.fields_to_export = ['uv_2d', 'elev_2d']
    options.fields_to_export_hdf5 = ['uv_2d', 'elev_2d']

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

    def export_func():
        options.export_func()
        i = solver_obj.i_export
        u_old, eta_old = solver_obj.timestepper.solution_old.split()
        u_old.rename("Velocity")
        eta_old.rename("Elevation")
        h5_dir = 'outputs/fixed_mesh/Velocity2d/hdf5'
        with DumbCheckpoint(os.path.join(h5_dir, f'Velocity2d_{index_str(i+1)}.h5'), mode=FILE_CREATE) as chk:
            chk.store(u_old)
        h5_dir = 'outputs/fixed_mesh/Elevation2d/hdf5'
        with DumbCheckpoint(os.path.join(h5_dir, f'Elevation2d_{index_str(i+1)}.h5'), mode=FILE_CREATE) as chk:
            chk.load(eta_old)

    # Solve forward on current subinterval
    solver_obj.iterate(update_forcings=update_forcings, export_func=export_func)
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


# Solve forward
num_meshes = 1
q = initial_condition(function_space)
control = Control(q)  # FIXME: gradient w.r.t. mixed function not correct
J = 0
dt = options.timestep
mesh_iteration_time = end_time/num_meshes
qoi = time_integrated_qoi
for i in range(num_meshes):
    t_start = i*mesh_iteration_time
    t_end = (i+1)*mesh_iteration_time
    q, J = solver(
        q, t_start, t_end, dt,
        J=J, qoi=lambda *args: assemble(qoi(*args)),
    )
print_output(f"Energy output = {J/3600000} kW h")

# Solve adjoint
pyadjoint.compute_gradient(J, control)
tape = pyadjoint.get_working_tape()
num_timesteps = int(end_time/dt)
solves_per_dt = 1
solve_blocks = [
    block for block in tape.get_blocks()
    if issubclass(block.__class__, GenericSolveBlock)
    and block.adj_sol is not None
][-num_timesteps*solves_per_dt::solves_per_dt]
outfile = File('outputs/fixed_mesh/Adjoint2d/Adjoint2d.pvd')
h5_dir = create_directory('outputs/fixed_mesh/Adjoint2d/hdf5')
dt_per_export = int(options.simulation_export_time/dt)
solves_per_export = solves_per_dt*dt_per_export
for i in range(len(solve_blocks)):
    if i % solves_per_export == 0:
        for j in range(2):
            z, zeta = solve_blocks[i+j].adj_sol.split()
            z.rename("Adjoint velocity")
            zeta.rename("Adjoint elevation")
            outfile.write(z, zeta)
            with DumbCheckpoint(os.path.join(h5_dir, f'adjoint_{i+j}.h5'), mode=FILE_CREATE) as chk:
                chk.store(z)
                chk.store(zeta)

# Evaluate error estimators
forward = Function(function_space)
u, eta = forward.split()
u.rename("Velocity")
eta.rename("Elevation")
forward_old = Function(function_space)
u_old, eta_old = forward_old.split()
u_old.rename("Velocity")
eta_old.rename("Elevation")
adjoint = Function(function_space)
z, zeta = adjoint.split()
z.rename("Adjoint velocity")
zeta.rename("Adjoint elevation")
adjoint_old = Function(function_space)
z_old, zeta_old = adjoint_old.split()
z_old.rename("Adjoint velocity")
zeta_old.rename("Adjoint elevation")
ee = ErrorEstimator(options, norm_type='L2')
outfile = File('outputs/fixed_mesh/ErrorEstimator2d/ErrorEstimator2d.pvd')
for i in range(len(solve_blocks)):
    if i % solves_per_export == 0:
        h5_dir = 'outputs/fixed_mesh/Velocity2d/hdf5'
        with DumbCheckpoint(os.path.join(h5_dir, f'Velocity2d_{index_str(i)}.h5'), mode=FILE_READ) as chk:
            chk.load(u_old)
        with DumbCheckpoint(os.path.join(h5_dir, f'Velocity2d_{index_str(i+1)}.h5'), mode=FILE_READ) as chk:
            chk.load(u)
        h5_dir = 'outputs/fixed_mesh/Elevation2d/hdf5'
        with DumbCheckpoint(os.path.join(h5_dir, f'Elevation2d_{index_str(i)}.h5'), mode=FILE_READ) as chk:
            chk.load(eta_old)
        with DumbCheckpoint(os.path.join(h5_dir, f'Elevation2d_{index_str(i+1)}.h5'), mode=FILE_READ) as chk:
            chk.load(eta)
        h5_dir = 'outputs/fixed_mesh/Adjoint2d/hdf5'
        with DumbCheckpoint(os.path.join(h5_dir, f'adjoint_{i}.h5'), mode=FILE_READ) as chk:
            chk.load(z_old)
            chk.load(zeta_old)
        with DumbCheckpoint(os.path.join(h5_dir, f'adjoint_{i+1}.h5'), mode=FILE_READ) as chk:
            chk.load(z)
            chk.load(zeta)
        dq = ee.difference_quotient(u, eta, u_old, eta_old, z, zeta, z_old, zeta_old)
        dq.rename("Difference quotient")
        outfile.write(dq)
