from thetis import *
from thetis.configuration import PositiveFloat, NonNegativeFloat, Bool
from thetis.options import ModelOptions2d
import numpy as np


__all__ = ["FarmOptions"]


class FarmOptions(ModelOptions2d):
    """
    Base class for parameters associated with
    tidal farm problems.
    """

    turbine_diameter = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(None, allow_none=True).tag(config=False)
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
        raise NotImplementedError("Should be implemented in derived class.")

    def apply_initial_conditions(self, solver_obj):
        raise NotImplementedError("Should be implemented in derived class.")

    @property
    def velocity_correction(self):
        """
        Correction to account for the fact that we are using an
        at-the-turbine velocity as opposed to an upstream velocity.

        See [Kramer and Piggott 2016], eq. (13).
        """
        if not self.correct_thrust:
            return 1.0
        c_T = self.thrust_coefficient
        D = self.turbine_diameter
        H = self.get_depth("turbine")
        swept_area = pi * (D / 2) ** 2
        return 0.5 * (1.0 + np.sqrt(1.0 - c_T * swept_area / (H * D)))

    @property
    def corrected_thrust_coefficient(self):
        """
        Correction to account for the fact that the thrust coefficient
        is based on an upstream velocity whereas we are using a depth
        averaged at-the-turbine velocity.

        See [Kramer and Piggott 2016], eq. (15).
        """
        return self.thrust_coefficient / self.velocity_correction ** 2

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
        W = self.turbine_width or self.turbine_diameter
        footprint_area = D * W
        kw = dict(domain=self.mesh2d)
        if not self.discrete_turbines:
            raise NotImplementedError("Continuous turbine method not supported")
        fo = TidalTurbineFarmOptions()
        fo.turbine_density = Constant(1.0 / footprint_area, **kw)
        fo.turbine_options.diameter = D
        fo.turbine_options.thrust_coefficient = self.corrected_thrust_coefficient
        fo.break_even_wattage = self.break_even_wattage
        self.tidal_turbine_farms = {farm_id: fo for farm_id in self.farm_ids}

    def check_mesh_reynolds_number(self):
        """
        Compute the mesh Reynolds number for the horizontal viscosity field.
        """
        nu = self.horizontal_viscosity
        mesh = nu.function_space().mesh()
        u = self.horizontal_velocity_scale
        P1 = get_functionspace(mesh, "CG", 1)
        Re_h = Function(P1, name="Reynolds number")
        # TODO: Would be better to use P0
        Re_h.project(mesh.delta_x * u / nu)
        Re_h_vec = Re_h.vector().gather()
        Re_h_min = Re_h_vec.min()
        Re_h_max = Re_h_vec.max()
        Re_h_mean = np.mean(Re_h_vec)
        print_output(f"min(Re_h)  = {Re_h_min:11.4e}")
        print_output(f"max(Re_h)  = {Re_h_max:11.4e}")
        print_output(f"mean(Re_h) = {Re_h_mean:11.4e}")
        return Re_h

    def enforce_mesh_reynolds_number(self):
        """
        Enforce the maximum mesh Reynolds number specified by
        :attr:`max_mesh_reynolds_number`.
        """
        nu = self.horizontal_viscosity
        Re_h = self.check_mesh_reynolds_number()
        Re_h_max = self.max_mesh_reynolds_number
        U = self.horizontal_velocity_scale
        if U is None:
            raise ValueError(
                "Cannot enforce mesh Reynolds number without characteristic velocity!"
            )
        print_output(f"Enforcing maximum mesh Reynolds number {Re_h_max:.2e}")
        h = nu.function_space().mesh().delta_x
        nu.interpolate(conditional(Re_h >= Re_h_max, h * U / Re_h_max, nu))

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
