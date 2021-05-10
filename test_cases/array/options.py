from turbine_adapt import *
from thetis.configuration import PositiveFloat, PositiveInteger


__all__ = ["ArrayOptions"]


class ArrayOptions(FarmOptions):
    """
    Parameters for the unsteady 15 turbine array test case from [Divett et al. 2013].
    """
    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(5.0).tag(config=False)
    array_length = PositiveInteger(5).tag(config=False)
    array_width = PositiveInteger(3).tag(config=False)
    num_turbines = PositiveInteger(15).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    def __init__(self, level=0, ramp_dir=None, meshgen=False, mesh=None, **kwargs):
        super(ArrayOptions, self).__init__()
        self.array_ids = np.array([[2, 5, 8, 11, 14],
                                   [3, 6, 9, 12, 15],
                                   [4, 7, 10, 13, 16]])
        self.farm_ids = tuple(self.array_ids.reshape((self.num_turbines, )))
        self.thrust_coefficient = 2.985

        # Domain and mesh
        self.ramp = None
        if mesh is None or ramp_dir is not None:
            self.mesh_file = os.path.join(self.resource_dir, f'channel_box_{level}.msh')
            if meshgen:
                return
            elif os.path.exists(self.mesh_file):
                self.mesh2d = Mesh(self.mesh_file)
            else:
                raise IOError("Need to make mesh before initialising ArrayOptions object.")
            if ramp_dir is not None:
                ramp_file = os.path.join(ramp_dir, 'ramp')
                if not os.path.exists(ramp_file + '.h5'):
                    raise IOError(f"No ramp file found at {ramp_file}")
                print_output(f"Using ramp file {ramp_file}.h5")
                V_2d = MixedFunctionSpace([
                    VectorFunctionSpace(self.mesh2d, "DG", 1, name="U_2d"),
                    get_functionspace(self.mesh2d, "DG", 1, name="H_2d"),
                ])
                self.ramp = Function(V_2d)
                uv, elev = self.ramp.split()
                with DumbCheckpoint(ramp_file, mode=FILE_READ) as chk:
                    chk.load(uv, name='uv_2d')
                    chk.load(elev, name='elev_2d')
        if mesh is not None:
            self.mesh2d = mesh

        # Physics
        self.depth = 50.0
        P1 = get_functionspace(self.mesh2d, "CG", 1)
        self.quadratic_drag_coefficient = Constant(0.0025)
        self.horizontal_velocity_scale = Constant(1.5)
        self.bathymetry2d = Function(P1, name='Bathymetry')
        self.bathymetry2d.assign(self.depth)
        self.base_viscosity = kwargs.get('base_viscosity', 10.0)
        self.target_viscosity = kwargs.get('target_viscosity', 0.01)
        self.max_mesh_reynolds_number = kwargs.get('max_mesh_reynolds_number', 1000.0)
        self.sponge_x = kwargs.get('sponge_x', 200.0)
        self.sponge_y = kwargs.get('sponge_y', 100.0)
        self.horizontal_viscosity = self.set_viscosity(P1)

        # Spatial discretisation
        self.use_lax_friedrichs_velocity = True
        self.use_grad_div_viscosity = False
        self.use_grad_depth_viscosity = True
        self.element_family = 'dg-dg'

        # Temporal discretisation
        self.timestep = 2.232
        while self.courant_number > 12:
            self.timestep /= 2
        print_output(f"Using timestep {self.timestep:.4f}s")
        self.tide_time = 0.1*self.M2_tide_period
        self.ramp_time = self.tide_time
        self.simulation_end_time = 2*self.tide_time
        self.simulation_export_time = 11.16

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.tide_time

        # I/O
        self.fields_to_export = ['uv_2d', 'elev_2d']
        self.fields_to_export_hdf5 = []

    def rebuild_mesh_dependent_components(self, mesh, **kwargs):
        """
        Rebuild all attributes which depend on :attr:`mesh2d`.
        """
        self.create_tidal_farm(mesh=mesh)
        P1 = get_functionspace(mesh, "CG", 1)
        self.bathymetry2d = Function(P1, name='Bathymetry')
        self.bathymetry2d.assign(self.depth)
        self.horizontal_viscosity = self.set_viscosity(P1)

    def set_viscosity(self, fs):
        """
        Set the viscosity to be the :attr:`target_viscosity` in the tidal farm region and
        :attr:`base_viscosity` elsewhere.
        """
        nu = Function(fs, name="Horizontal viscosity")

        # Get box around tidal farm
        D = self.turbine_diameter
        delta_x = 3*10*D
        delta_y = 1.3*7.5*D

        # Base viscosity and minimum viscosity
        nu_tgt = self.target_viscosity
        nu_base = self.base_viscosity
        if np.isclose(nu_tgt, nu_base):
            nu.assign(nu_base)
            return nu

        # Distance functions
        x, y = SpatialCoordinate(self.mesh2d)
        dist_x = (abs(x) - delta_x)/self.sponge_x
        dist_y = (abs(y) - delta_y)/self.sponge_y
        dist_r = sqrt(dist_x**2 + dist_y**2)

        # Define viscosity field with a sponge condition
        nu.interpolate(
            conditional(
                And(x > -delta_x, x < delta_x),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    nu_tgt,
                    min_value(nu_tgt*(1 - dist_y) + nu_base*dist_y, nu_base),
                ),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    min_value(nu_tgt*(1 - dist_x) + nu_base*dist_x, nu_base),
                    min_value(nu_tgt*(1 - dist_r) + nu_base*dist_r, nu_base),
                ),
            )
        )

        # Enforce maximum Reynolds number
        Re_h, Re_h_min, Re_h_max = self.check_mesh_reynolds_number(nu)
        if Re_h_max > self.max_mesh_reynolds_number:
            nu_enforced = self.enforce_mesh_reynolds_number(nu)
            nu.interpolate(conditional(Re_h > self.max_mesh_reynolds_number, nu_enforced, nu))
        return nu

    def get_bnd_conditions(self, V_2d):
        self.elev_in = Function(V_2d.sub(1))
        self.elev_out = Function(V_2d.sub(1))
        self.bnd_conditions = {
            1: {'un': Constant(0.0)},    # bottom
            2: {'elev': self.elev_out},  # right
            3: {'un': Constant(0.0)},    # top
            4: {'elev': self.elev_in},   # left
        }

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        self.get_bnd_conditions(solver_obj.function_spaces.V_2d)
        solver_obj.bnd_functions['shallow_water'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        """
        For spin-up, a small, non-zero velocity which
        satisfies the free-slip conditions is assumed, along
        with a linear elevation field which satisfies the
        forced boundary conditions. Otherwise, the spun-up
        solution field is loaded from file.
        """
        q = Function(solver_obj.function_spaces.V_2d)
        u, eta = q.split()

        if self.ramp is not None:
            u_ramp, eta_ramp = self.ramp.split()
            u.project(u_ramp)
            eta.project(eta_ramp)
        else:
            u.interpolate(as_vector([1e-8, 0.0]))
            x, y = SpatialCoordinate(self.mesh2d)
            X = 2*x/self.domain_length  # Non-dimensionalised x
            eta.interpolate(-self.max_amplitude*X)

        solver_obj.assign_initial_conditions(uv=u, elev=eta)

    def get_update_forcings(self):
        """
        Simple tidal forcing with frequency :attr:`omega` and amplitude :attr:`max_amplitude`.
        """
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)
        ramped = self.ramp is not None

        def update_forcings(t):
            tc.assign(t + self.ramp_time if ramped else t)
            self.elev_in.assign(hmax*cos(self.omega*tc))
            self.elev_out.assign(hmax*cos(self.omega*tc + pi))

        return update_forcings
