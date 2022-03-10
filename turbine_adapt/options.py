from turbine_adapt import *
from thetis.configuration import PositiveFloat, NonNegativeFloat, Bool
from thetis.options import ModelOptions2d


__all__ = ["FarmOptions"]


class FarmOptions(ModelOptions2d):
    """
    Base class for parameters associated with
    tidal farm problems.
    """

    sea_water_density = PositiveFloat(
        1030.0,
        help="""
        Density of sea water in kg m^{-3}.

        This is used when computing power and
        energy output.
        """,
    ).tag(config=True)
    M2_tide_period = PositiveFloat(
        12.4 * 3600,
        help="""
        Period of the M2 tidal constituent in
        seconds.
        """,
    ).tag(config=False)
    gravitational_acceleration = PositiveFloat(
        9.81,
        help="""
        Gravitational acceleration in m s^{-2}.
        """,
    ).tag(config=False)
    thrust_coefficient = PositiveFloat(
        0.8,
        help="""
        Uncorrected dimensionless drag associated
        with a turbine.
        """,
    ).tag(config=True)
    correct_thrust = Bool(
        True,
        help="""
        Toggle whether to apply the thrust correction
        recommended in [Kramer and Piggott 2016].
        """,
    ).tag(config=True)
    max_mesh_reynolds_number = PositiveFloat(
        1000.0,
        help="""
        Maximum tolerated mesh Reynolds number.
        """,
    ).tag(config=True)
    target_mesh_reynolds_number = PositiveFloat(
        None,
        help="""
        Target mesh Reynolds number.
        """,
        allow_none=True,
    ).tag(config=True)
    break_even_wattage = NonNegativeFloat(
        0.0,
        help="""
        Minimum Wattage to be reached before activating a
        turbine.
        """,
    ).tag(config=True)
    discrete_turbines = Bool(
        True,
        help="""
        Toggle whether to consider turbines as indicator
        functions over their footprints (discrete) or as
        a density field (continuous).
        """,
    ).tag(config=True)

    def __init__(self, **kwargs):
        super(FarmOptions, self).__init__()
        self._isfrozen = False
        self.update(kwargs)

    def get_update_forcings(self):
        return lambda t: None

    @property
    def update_forcings(self):
        return self.get_update_forcings()

    def get_export_func(self):
        return lambda: None

    @property
    def export_func(self):
        return self.get_export_func()

    def get_bnd_conditions(self, V_2d):
        self.bnd_conditions = {}

    def apply_boundary_conditions(self, solver_obj):
        """
        Should be implemented in derived class.
        """
        pass

    def apply_initial_conditions(self, solver_obj):
        """
        Should be implemented in derived class.
        """
        pass

    @property
    def corrected_thrust_coefficient(self):
        """
        Correction to account for the fact that the thrust coefficient
        is based on an upstream velocity whereas we are using a depth
        averaged at-the-turbine velocity (see [Kramer and Piggott 2016],
        eq. (15)).
        """
        c_T = self.thrust_coefficient
        if not self.correct_thrust:
            return c_T
        D = self.turbine_diameter
        H = self.get_depth("turbine")
        swept_area = pi * (D / 2) ** 2
        cross_sectional_area = H * D
        correction = (
            4.0 / (1.0 + np.sqrt(1.0 - c_T * swept_area / cross_sectional_area)) ** 2
        )
        return c_T * correction

    def create_tidal_farm(self, mesh=None):
        """
        Associate a tidal farm object with each farm ID, so that the power
        output of each turbine can be computed independently.
        """
        if mesh is not None:
            self.mesh2d = mesh
        assert hasattr(self, "turbine_diameter")
        assert hasattr(self, "farm_ids")
        assert len(self.farm_ids) > 0
        D = self.turbine_diameter
        W = (
            self.turbine_diameter
            if not hasattr(self, "turbine_width")
            else self.turbine_width
        )
        footprint_area = D * W
        farm_options = TidalTurbineFarmOptions()
        if self.discrete_turbines:
            farm_options.turbine_density = Constant(
                1.0 / footprint_area, domain=self.mesh2d
            )
        else:
            # NOTE: Assumes turbines aligned with xy-axes with
            #       D and W corresponding to x and y axes, resp.
            x, y = SpatialCoordinate(self.mesh2d)
            farm_options.turbine_density = 0
            for tag in self.farm_ids:
                area = assemble(Constant(1.0, domain=self.mesh2d) * dx(tag))
                _x, _y = assemble(x * dx(tag)) / area, assemble(y * dx(tag)) / area
                q0, q1 = (x - _x) ** 2 / D**2, (y - _y) ** 2 / W**2
                bump = conditional(
                    And(lt(q0, 1), lt(q1, 1)),
                    exp(1 - 1 / (1 - q0)) * exp(1 - 1 / (1 - q1)),
                    0,
                )
                area = assemble(bump * dx)
                farm_options.turbine_density = (
                    farm_options.turbine_density + bump / area
                )
        farm_options.turbine_options.diameter = D
        farm_options.turbine_options.thrust_coefficient = (
            self.corrected_thrust_coefficient
        )
        farm_options.break_even_wattage = self.break_even_wattage
        self.tidal_turbine_farms = {farm_id: farm_options for farm_id in self.farm_ids}

    def check_mesh_reynolds_number(self, nu=None, mesh=None):
        """
        Compute the mesh Reynolds number for a given viscosity field.
        """
        nu = nu or self.horizontal_viscosity
        if isinstance(nu, Constant) and np.isclose(nu.values()[0], 0.0):
            print_output("Cannot compute mesh Reynolds number for inviscid problems!")
            return 3 * [None]
        mesh = mesh or nu.function_space().mesh()
        u = self.horizontal_velocity_scale
        fs = (
            get_functionspace(mesh, "CG", 1)
            if isinstance(nu, Constant)
            else nu.function_space()
        )
        Re_h = Function(fs, name="Reynolds number")
        Re_h.project(mesh.delta_x * u / nu)
        # Re_h.interpolate(mesh.delta_x*u/nu)
        Re_h_vec = Re_h.vector().gather()
        Re_h_min = Re_h_vec.min()
        Re_h_max = Re_h_vec.max()
        Re_h_mean = np.mean(Re_h_vec)
        lg = lambda x: "<" if x < 1 else ">"
        print_output("min(Re_h)  = {:11.4e} {:1s} 1".format(Re_h_min, lg(Re_h_min)))
        print_output("max(Re_h)  = {:11.4e} {:1s} 1".format(Re_h_max, lg(Re_h_max)))
        print_output("mean(Re_h) = {:11.4e} {:1s} 1".format(Re_h_mean, lg(Re_h_mean)))
        return Re_h, Re_h_min, Re_h_max

    def enforce_mesh_reynolds_number(self, nu):
        """
        Enforce the mesh Reynolds number specified by :attr:`max_mesh_reynolds_number`.
        """
        Re_h = self.max_mesh_reynolds_number
        if Re_h is None:
            raise ValueError(
                "Cannot enforce mesh Reynolds number if it isn't specified!"
            )
        u = self.horizontal_velocity_scale
        if u is None:
            raise ValueError(
                "Cannot enforce mesh Reynolds number without characteristic velocity!"
            )
        print_output("Enforcing mesh Reynolds number {:.4e}...".format(Re_h))

        # Compute viscosity which yields target mesh Reynolds number
        _nu = Function(nu, name="Modified viscosity")
        _nu.project(self.mesh2d.delta_x * u / Re_h)
        _nu.interpolate(max_value(_nu, 0.0))
        return _nu

    def get_depth(self, mode=None):
        if mode is None:
            assert hasattr(self, "depth")
            return self.depth
        elif mode == "turbine":
            if hasattr(self, "depth"):
                return self.depth
            elif hasattr(self, "turbine_depth"):
                return self.turbine_depth
            elif hasattr(self, "min_depth"):
                return self.min_depth
            elif hasattr(self, "bathymetry2d"):
                return self.bathymetry2d.vector().gather().min()
            else:
                raise ValueError("Cannot deduce maximum depth")
        else:
            assert mode == "max", f"Unrecognised mode {mode}"
            if hasattr(self, "depth"):
                return self.depth
            elif hasattr(self, "max_depth"):
                return self.max_depth
            elif hasattr(self, "bathymetry2d"):
                return self.bathymetry2d.vector().gather().max()
            else:
                raise ValueError("Cannot deduce maximum depth")

    @property
    def courant_number(self):
        """
        Compute the Courant number based on maximum depth and minimum element spacing.
        """
        H = self.get_depth("max")
        celerity = np.sqrt(self.gravitational_acceleration * H)
        delta_x = self.mesh2d.delta_x.vector().gather().min()
        return celerity * self.timestep / delta_x
