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

    def __init__(self, level=0, meshgen=False, mesh=None, **kwargs):
        super(ArrayOptions, self).__init__()
        self.array_ids = np.array([[2, 5, 8, 11, 14],
                                   [3, 6, 9, 12, 15],
                                   [4, 7, 10, 13, 16]])
        self.farm_ids = tuple(self.array_ids.reshape((self.num_turbines, )))
        self.thrust_coefficient = 2.985

        # Domain and mesh
        if mesh is not None:
            self.mesh2d = mesh
        else:
            self.mesh_file = os.path.join(self.resource_dir, 'channel_box_{:d}.msh'.format(level))
            if meshgen:
                return
            elif os.path.exists(self.mesh_file):
                self.mesh2d = Mesh(self.mesh_file)
            else:
                raise IOError("Need to make mesh before initialising ArrayOptions object.")

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
        print_output("Using timestep {:.4f}s".format(self.timestep))
        self.tide_time = 0.1*self.M2_tide_period
        self.simulation_end_time = 2*self.tide_time
        self.simulation_export_time = 11.16

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2*pi/self.tide_time

        # I/O
        self.fields_to_export = ['uv_2d', 'elev_2d']
        self.fields_to_export_hdf5 = []

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

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        V_2d = solver_obj.function_spaces.V_2d
        self.elev_in = Function(V_2d.sub(1))
        self.elev_out = Function(V_2d.sub(1))
        inflow_tag = 4
        outflow_tag = 2
        solver_obj.bnd_functions['shallow_water'] = {
            inflow_tag: {'elev': self.elev_in},
            outflow_tag: {'elev': self.elev_out},
        }

    def apply_initial_conditions(self, solver_obj):
        """
        Specify elevation so that it satisfies the boundary
        forcing and set an arbitrary small velocity.
        """
        q = Function(solver_obj.function_spaces.V_2d)
        u, eta = q.split()

        # Set an arbitrary, small, non-zero velocity which satisfies the free-slip conditions
        u.interpolate(as_vector([1e-8, 0.0]))

        # Set the initial surface so that it satisfies the forced boundary conditions
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

        def update_forcings(t):
            tc.assign(t)
            self.elev_in.assign(hmax*cos(self.omega*tc))
            self.elev_out.assign(hmax*cos(self.omega*tc + pi))

        return update_forcings
