from turbine_adapt import *
from thetis.configuration import PositiveFloat, PositiveInteger


__all__ = ["TwoTurbineOptions"]


class TwoTurbineOptions(FarmOptions):
    """
    Parameters for the 2 turbine problem
    """
    mesh_dir = os.path.join(os.path.dirname(__file__), 'resources', 'meshes')

    # Farm parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    array_length = PositiveInteger(2).tag(config=False)
    array_width = PositiveInteger(1).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(1200.0).tag(config=False)
    domain_width = PositiveFloat(500.0).tag(config=False)

    def __init__(self, level=1, meshgen=False, box=True, **kwargs):
        """
        :kwarg level: number of iso-P2 refinements to apply to the base mesh.
        :kwarg offset: offset of the turbines to the south and north in terms of turbine diameters.
        :kwarg separation: number of turbine diameters separating the two.
        :kwarg generate_geo: if True, the mesh is not built (used in meshgen.py).
        """
        self.offset = kwargs.pop('offset', 0)
        self.separation = kwargs.pop('separation', 8)
        nu = kwargs.pop('viscosity', 0.5)
        super(TwoTurbineOptions, self).__init__(**kwargs)
        self.array_ids = np.array([2, 3])
        self.farm_ids = (2, 3)

        # Turbine geometries
        D = self.turbine_diameter
        L = self.domain_length
        W = self.domain_width
        S = self.separation
        yloc = [W/2, W/2]
        yloc[0] -= self.offset*D
        yloc[1] += self.offset*D
        self.turbine_geometries = [(L/2-S*D, yloc[0], D/2), (L/2+S*D, yloc[1], D/2)]
        assert len(self.turbine_geometries) == self.num_turbines

        # Domain and mesh
        self.base_outer_res = 30.0
        self.base_inner_res = 8.0
        if box:
            assert self.offset == 0  # TODO
            self.mesh_file = 'channel_refined_{:d}.msh'.format(level)
        else:
            self.mesh_file = 'channel_{:d}_{:d}.msh'.format(level, self.offset)
        self.mesh_file = os.path.join(self.mesh_dir, self.mesh_file)
        if meshgen:
            return
        self.mesh2d = Mesh(self.mesh_file)
        P1_2d = get_functionspace(self.mesh2d, "CG", 1)

        # Physics
        self.depth = 40.0  # Typical depth in Pentland Firth
        self.horizontal_velocity_scale = Constant(5.0)  # Typical fast flow in Pentland Firth
        self.inflow_velocity = as_vector([self.horizontal_velocity_scale.values()[0], 0.0])
        self.horizontal_viscosity = Constant(nu)
        self.bathymetry2d = Function(P1_2d)
        self.bathymetry2d.assign(self.depth)
        self.quadratic_drag_coefficient = Constant(0.0025)
        self.flow_speed_ramped = Constant(0.0)

        # Spatial discretisation
        self.element_family = 'dg-dg'
        self.use_lax_friedrichs_velocity = True
        self.grad_div_viscosity = False
        self.grad_depth_viscosity = False

        # Temporal discretisation
        self.simulation_export_time = 10.0
        self.simulation_end_time = 3600.0
        # self.timestepper_type = 'DIRK22'
        self.timestepper_type = 'CrankNicolson'
        self.timestep = 2.0
        self.ramp_time = 300.0

        # I/O
        self.fields_to_export = ['uv_2d']
        self.fields_to_export_hdf5 = []

    def apply_boundary_conditions(self, solver_obj):
        flux = Constant(self.depth*self.domain_width)*self.flow_speed_ramped
        inflow_tag = 1
        outflow_tag = 2
        solver_obj.bnd_functions['shallow_water'] = {
            inflow_tag: {'flux': -flux, 'elev': Constant(0.0)},
            outflow_tag: {'flux': flux, 'elev': Constant(0.0)},
        }

    def apply_initial_conditions(self, solver_obj):
        solver_obj.assign_initial_conditions(uv=Constant(as_vector([1.0e-04, 0.0])))

    def get_update_forcings(self):
        tc = Constant(0.0)
        u_ramp = self.horizontal_velocity_scale*conditional(
            le(tc, self.ramp_time),
            tc/self.ramp_time,
            Constant(1.0)
        )

        def update_forcings(t):
            tc.assign(t)
            self.flow_speed_ramped.assign(u_ramp)

        return update_forcings
