from turbine_adapt import *
from thetis.configuration import PositiveFloat, PositiveInteger


__all__ = ["SpaceshipOptions"]


class SpaceshipOptions(FarmOptions):
    """
    Parameters for the 'spaceship' test case from [Walkington & Burrows 2009].
    """
    resource_dir = os.path.join(os.path.dirname(__file__), 'resources')

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    array_length = PositiveInteger(2).tag(config=False)
    array_width = PositiveInteger(1).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(61500.0).tag(config=False)
    domain_width = PositiveFloat(60000.0).tag(config=False)
    narrows_width = PositiveFloat(1000.0).tag(config=False)
    maximum_upstream_width = PositiveFloat(5000.0).tag(config=False)

    def __init__(self, mesh=None, **kwargs):
        super(SpaceshipOptions, self).__init__(**kwargs)
        self.array_ids = np.array([3, 2])
        self.farm_ids = tuple(self.array_ids)
        self.thrust_coefficient = 0.8
        self.ramp_dir = kwargs.get('ramp_dir')

        # Domain and mesh
        if mesh is None:
            self.mesh_file = os.path.join(self.resource_dir, 'spaceship.msh')
            if os.path.exists(self.mesh_file):
                self.mesh2d = Mesh(self.mesh_file)
            else:
                raise IOError("Need to make mesh before initialising SpaceshipOptions object.")
        else:
            self.mesh2d = mesh

        # Physics
        self.max_depth = 25.5
        self.turbine_depth = 25.5
        P1 = get_functionspace(self.mesh2d, "CG", 1)
        self.quadratic_drag_coefficient = Constant(0.0025)
        self.set_bathymetry(P1)
        self.base_viscosity = 5.0
        self.max_viscosity = 1000.0
        self.viscosity_sponge_type = kwargs.get('viscosity_sponge_type', 'linear')
        self.set_viscosity(P1)

        # Spatial discretisation
        self.use_lax_friedrichs_velocity = False
        self.use_grad_div_viscosity = False
        self.use_grad_depth_viscosity = True
        self.element_family = 'dg-cg'

        # Temporal discretisation
        self.timestep = 10.0
        self.tide_time = self.M2_tide_period
        self.ramp_time = 20.4*3600.0
        self.simulation_end_time = 3*24*3600.0
        self.simulation_export_time = 300.0
        # self.timestepper = 'CrankNicolson'
        self.timestepper = 'PressureProjectionPicard'
        self.timestepper_options.implicitness_theta = 1.0
        self.timestepper_options.use_semi_implicit_linearization = True

        # I/O
        self.fields_to_export = ['uv_2d', 'elev_2d']
        self.fields_to_export_hdf5 = []

    @property
    def ramp(self):
        if self.ramp_dir is None:
            return
        ramp_file = os.path.join(self.ramp_dir, "ramp")
        if not os.path.exists(ramp_file + '.h5'):
            raise IOError(f"No ramp file found at {ramp_file}.h5")
        print_output(f"Using ramp file {ramp_file}.h5")
        element = ("DG", 1) if self.element_family == 'dg-dg' else ("CG", 2)
        ramp = Function(MixedFunctionSpace([
            VectorFunctionSpace(self.mesh2d, "DG", 1, name="U_2d"),
            get_functionspace(self.mesh2d, element, name="H_2d"),
        ]))
        uv, elev = ramp.split()
        with DumbCheckpoint(ramp_file, mode=FILE_READ) as chk:
            chk.load(uv, name='uv_2d')
            chk.load(elev, name='elev_2d')
        return ramp

    def rebuild_mesh_dependent_components(self, mesh, **kwargs):
        """
        Rebuild all attributes which depend on :attr:`mesh2d`.
        """
        self.create_tidal_farm(mesh=mesh)

        P1 = get_functionspace(mesh, "CG", 1)
        self.set_bathymetry(P1)
        self.set_viscosity(P1)

    @property
    def tidal_forcing_interpolator(self):
        """
        Read tidal forcing data from the 'forcing.dat' file in the resource directory using the
        method :attr:`extract_data` and create a 1D linear interpolator.

        As a side-product, we determine the maximum amplitude of the tidal forcing and also the
        time period within which these data are available.
        """
        import scipy.interpolate as si

        if not hasattr(self, '_tidal_forcing_interpolator'):
            data_file = os.path.join(self.resource_dir, 'forcing.dat')
            if not os.path.exists(data_file):
                raise IOError(f"Tidal forcing data cannot be found in {self.resource_dir}.")
            times, data = [], []
            with open(data_file, 'r') as f:
                for line in f:
                    time, dat = line.split()
                    times.append(float(time))
                    data.append(float(dat))
            self.max_amplitude = np.max(np.abs(data))
            self.tidal_forcing_end_time = times[-1]
            self._tidal_forcing_interpolator = si.interp1d(times, data)
        return self._tidal_forcing_interpolator

    def set_bathymetry(self, fs):
        x, y = SpatialCoordinate(self.mesh2d)
        self.bathymetry2d = Function(fs)
        x1, x2 = 20000, 31500
        y1, y2 = 25.5, 4.5
        self.bathymetry2d.interpolate(min_value(((x - x1)*(y2 - y1)/(x2 - x1) + y1), y1))

    def set_viscosity(self, fs):
        """
        We use a sponge condition on the forced boundary.

        The type of sponge condition is specified by :attr:`viscosity_sponge_type`, which may be
        None, or chosen from {'linear', 'exponential'}. The sponge ramps up the viscosity from
        :attr:`base_viscosity` to :attr:`max_viscosity`.
        """
        self.horizontal_viscosity = Function(fs, name="Horizontal viscosity")
        x, y = SpatialCoordinate(fs.mesh())
        R = 30000.0  # Radius of semicircular part of domain
        r = sqrt(x**2 + y**2)/R
        base_viscosity = self.base_viscosity
        if self.viscosity_sponge_type is None:
            return base_viscosity
        if self.viscosity_sponge_type == 'linear':
            sponge = base_viscosity + r*(self.max_viscosity - base_viscosity)
        elif self.viscosity_sponge_type == 'exponential':
            sponge = base_viscosity + (exp(r) - 1)/(e - 1)*(self.max_viscosity - base_viscosity)
        else:
            msg = "Viscosity sponge type {:s} not recognised."
            raise ValueError(msg.format(self.viscosity_sponge_type))
        self.horizontal_viscosity.interpolate(max_value((x <= 0.0)*sponge, base_viscosity))

    def apply_boundary_conditions(self, solver_obj):
        self.elev_in = Constant(0.0)
        self.bnd_conditions = {
            1: {'un': Constant(0.0)},
            2: {'elev': self.elev_in},
        }
        solver_obj.bnd_functions['shallow_water'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        u, eta = Function(solver_obj.function_spaces.V_2d).split()

        if self.ramp_dir is None:
            u.interpolate(as_vector([1.0e-08, 0.0]))
        else:
            u_ramp, eta_ramp = self.ramp.split()
            u.project(u_ramp)
            eta.project(eta_ramp)

        solver_obj.assign_initial_conditions(uv=u, elev=eta)

    def get_update_forcings(self):
        """
        Interpolator for the tidal forcing data.
        """
        assert hasattr(self, 'elev_in'), "First apply boundary conditions"
        ramped = self.ramp_dir is not None

        def update_forcings(t):
            tau = t + self.ramp_time if ramped else t
            self.elev_in.assign(float(self.tidal_forcing_interpolator(tau)))

        return update_forcings
