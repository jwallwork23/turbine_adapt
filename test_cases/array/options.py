from thetis import *
from thetis.configuration import PositiveFloat, PositiveInteger
from turbine_adapt.options import FarmOptions
from pyroteus.utility import Mesh
import numpy as np
import os


__all__ = ["ArrayOptions"]


class ArrayOptions(FarmOptions):
    """
    Parameters for the unsteady 15 turbine array test case from [Divett et al. 2013].
    """

    resource_dir = os.path.join(os.path.dirname(__file__), "resources")

    # Turbine parameters
    turbine_diameter = PositiveFloat(20.0).tag(config=False)
    turbine_width = PositiveFloat(5.0).tag(config=False)
    array_length = PositiveInteger(5).tag(config=False)
    array_width = PositiveInteger(3).tag(config=False)
    num_turbines = PositiveInteger(15).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(3000.0).tag(config=False)
    domain_width = PositiveFloat(1000.0).tag(config=False)

    def __init__(self, configuration="aligned", meshgen=False, mesh=None, uniform=False, **kwargs):
        super(ArrayOptions, self).__init__()
        self.array_ids = np.array(
            [[2, 5, 8, 11, 14], [3, 6, 9, 12, 15], [4, 7, 10, 13, 16]]
        )
        self.column_ids = self.array_ids.transpose()
        self.farm_ids = tuple(self.column_ids.flatten())
        self.thrust_coefficient = 2.985
        self.ramp_level = kwargs.get("ramp_level", 0)
        self.uniform = uniform
        self.spunup = kwargs.get("spunup", True)
        self.configuration = configuration
        self.use_automatic_timestep = kwargs.get("use_automatic_timestep", False)
        self.max_courant_number = kwargs.get("max_courant_number", 10)
        self.ramp_dir = kwargs.get("ramp_dir")
        if self.ramp_dir is None:
            fpath = f"{os.path.abspath(os.path.dirname(__file__))}/outputs/{configuration}"
            approach = "uniform_mesh" if uniform else "fixed_mesh"
            self.ramp_dir = f"{fpath}/{approach}/level{self.ramp_level}/ramp1/hdf5"

        # Temporal discretisation
        self.timestep = 2.232
        if self.use_automatic_timestep:
            while self.courant_number > self.max_courant_number:
                self.timestep /= 2
        self.tide_time = 0.1 * self.M2_tide_period
        self.ramp_time = self.tide_time
        self.simulation_end_time = 0.5 * self.tide_time
        self.simulation_export_time = 11.16

        # Domain and mesh
        if mesh is None:
            level = kwargs.get("level", 0)
            fpath = f"{self.resource_dir}/{configuration}"
            label = "uniform" if uniform else "box"
            self.mesh_file = f"{fpath}/channel_{label}_{level}.msh"
            if meshgen:
                return
            elif os.path.exists(self.mesh_file):
                self.mesh2d = Mesh(self.mesh_file)
            else:
                raise IOError(
                    "Need to make mesh before initialising ArrayOptions object."
                )
        else:
            self.mesh2d = mesh

        # Physics
        self.depth = 50.0
        P1 = get_functionspace(self.mesh2d, "CG", 1)
        self.quadratic_drag_coefficient = Constant(0.0025)
        self.horizontal_velocity_scale = Constant(4.0)
        self.bathymetry2d = Function(P1, name="Bathymetry")
        self.bathymetry2d.assign(self.depth)
        self.base_viscosity = kwargs.get("base_viscosity", 10.0)
        self.target_viscosity = kwargs.get("target_viscosity", 0.01)
        self.max_mesh_reynolds_number = kwargs.get("max_mesh_reynolds_number", 1000.0)
        self.sponge_x = kwargs.get("sponge_x", 200.0)
        self.sponge_y = kwargs.get("sponge_y", 100.0)
        self.set_viscosity(P1)

        # Spatial discretisation
        self.use_lax_friedrichs_velocity = True
        self.use_grad_div_viscosity = False
        self.use_grad_depth_viscosity = True
        self.element_family = "dg-dg"

        # Solver parameters
        self.swe_timestepper_options.solver_parameters = {
            "snes_type": "newtonls",
            "ksp_type": "gmres",
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
        }

        # Boundary forcing
        self.max_amplitude = 0.5
        self.omega = 2 * pi / self.tide_time

        # I/O
        self.fields_to_export = kwargs.get("fields_to_export", ["uv_2d"])
        self.fields_to_export_hdf5 = kwargs.get("fields_to_export_hdf5", [])

    def ramp(self, fs=None):
        """
        Load spun-up state from file.

        :kwarg fs: if not ``None``, the spun-up state will be
            projected into this :class:`FunctionSpace`.
        """
        idx = int(np.round(self.ramp_time/self.simulation_export_time))
        ramp_file = f"{self.ramp_dir}/Velocity2d_{idx:05d}"
        if fs is None:
            label = "uniform" if "uniform" in self.ramp_dir else "box"
            fpath = f"{self.resource_dir}/{self.configuration}"
            mesh2d = Mesh(f"{fpath}/channel_{label}_{self.ramp_level}.msh")
            fs = get_functionspace(mesh2d, "DG", 1, vector=True) * get_functionspace(mesh2d, "DG", 1)
        ramp = Function(fs)
        uv, elev = ramp.split()
        if not os.path.exists(ramp_file + ".h5"):
            raise IOError(f"No ramp file found at {ramp_file}.h5")
        print_output(f"Using velocity ramp file {ramp_file}.h5")
        with DumbCheckpoint(ramp_file, mode=FILE_READ) as chk:
            chk.load(uv, name="uv_2d")
        ramp_file = f"{self.ramp_dir}/Elevation2d_{idx:05d}"
        if not os.path.exists(ramp_file + ".h5"):
            raise IOError(f"No ramp file found at {ramp_file}.h5")
        print_output(f"Using elevation ramp file {ramp_file}.h5")
        with DumbCheckpoint(ramp_file, mode=FILE_READ) as chk:
            chk.load(elev, name="elev_2d")
        return ramp

    def rebuild_mesh_dependent_components(self, mesh, **kwargs):
        """
        Rebuild all attributes which depend on :attr:`mesh2d`.
        """
        self.create_tidal_farm(mesh=mesh)

        P1 = get_functionspace(mesh, "CG", 1)
        self.bathymetry2d = Function(P1, name="Bathymetry")
        self.bathymetry2d.assign(self.depth)
        self.set_viscosity(P1)

        self.timestep = 2.232
        if self.use_automatic_timestep:
            while self.courant_number > self.max_courant_number:
                self.timestep /= 2

    def set_viscosity(self, fs):
        """
        Set the viscosity to be the :attr:`target_viscosity` in the tidal farm region and
        :attr:`base_viscosity` elsewhere.
        """
        self.horizontal_viscosity = Function(fs, name="Horizontal viscosity")
        target = self.target_viscosity
        base = self.base_viscosity

        # Get box around tidal farm
        delta_x = 1000.0
        delta_y = 250.0

        # Distance functions
        x, y = SpatialCoordinate(self.mesh2d)
        dist_x = (abs(x) - delta_x) / self.sponge_x
        dist_y = (abs(y) - delta_y) / self.sponge_y
        dist_r = sqrt(dist_x**2 + dist_y**2)

        # Define viscosity field with a sponge condition
        self.horizontal_viscosity.interpolate(
            conditional(
                And(x > -delta_x, x < delta_x),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    target,
                    min_value(target * (1 - dist_y) + base * dist_y, base),
                ),
                conditional(
                    And(y > -delta_y, y < delta_y),
                    min_value(target * (1 - dist_x) + base * dist_x, base),
                    min_value(target * (1 - dist_r) + base * dist_r, base),
                ),
            )
        )

        # Enforce maximum Reynolds number
        self.enforce_mesh_reynolds_number()
        self.check_mesh_reynolds_number()

    def get_bnd_conditions(self, V_2d):
        self.elev_in = Function(V_2d.sub(1))
        self.elev_out = Function(V_2d.sub(1))
        self.bnd_conditions = {
            1: {"un": Constant(0.0)},  # bottom
            2: {"elev": self.elev_out},  # right
            3: {"un": Constant(0.0)},  # top
            4: {"elev": self.elev_in},  # left
        }

    def apply_boundary_conditions(self, solver_obj):
        if len(solver_obj.function_spaces.keys()) == 0:
            solver_obj.create_function_spaces()
        self.get_bnd_conditions(solver_obj.function_spaces.V_2d)
        solver_obj.bnd_functions["shallow_water"] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        """
        For spin-up, a small, non-zero velocity which
        satisfies the free-slip conditions is assumed, along
        with a linear elevation field which satisfies the
        forced boundary conditions. Otherwise, the spun-up
        solution field is loaded from file.
        """
        fs = solver_obj.function_spaces.V_2d
        u, eta = Function(fs).split()

        if self.spunup:
            u_ramp, eta_ramp = self.ramp(fs).split()
            u.project(u_ramp)
            eta.project(eta_ramp)
        else:
            u.interpolate(as_vector([1.0e-08, 0.0]))
            x, y = SpatialCoordinate(self.mesh2d)
            X = 2 * x / self.domain_length  # Non-dimensionalised x
            eta.interpolate(-self.max_amplitude * X)

        solver_obj.assign_initial_conditions(uv=u, elev=eta)

    def get_update_forcings(self):
        """
        Simple tidal forcing with frequency :attr:`omega` and amplitude :attr:`max_amplitude`.
        """
        if not hasattr(self, "elev_in") or not hasattr(self, "elev_out"):
            raise ValueError("First apply boundary conditions")
        tc = Constant(0.0)
        hmax = Constant(self.max_amplitude)

        def update_forcings(t):
            tc.assign(t + self.ramp_time if self.spunup else t)
            self.elev_in.assign(hmax * cos(self.omega * tc))
            self.elev_out.assign(hmax * cos(self.omega * tc + pi))

        return update_forcings
