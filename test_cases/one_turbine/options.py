from turbine_adapt import *
from thetis.configuration import PositiveFloat, PositiveInteger


__all__ = ["TurbineOptions"]


class TurbineOptions(FarmOptions):
    """
    Parameters for a simple test case involving a single
    turbine. The test case is based on the flow around a
    cylinder test case in Thetis:
        thetis/examples/cylinder_eddies/cylinder_eddies.py
    """
    self.mesh_file = os.path.join(os.path.dirname(__file__), 'turbine.msh')

    # Turbine parameters
    turbine_diameter = PositiveFloat(500.0).tag(config=False)
    array_length = PositiveInteger(1).tag(config=False)
    array_width = PositiveInteger(1).tag(config=False)
    num_turbines = PositiveInteger(1).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(19000.0).tag(config=False)
    domain_width = PositiveFloat(7000.0).tag(config=False)

    def __init__(self):
        super(TurbineOptions, self).__init__()
        self.array_ids = np.array([2])
        self.farm_ids = (2, )

        # Domain and mesh
        if os.path.exists(self.mesh_file):
            self.mesh2d = Mesh(self.mesh_file)
        else:
            raise IOError("Mesh '{:s}' does not exist.".format(self.mesh_file))

        # Physics
        self.depth = 20.0
        P1_2d = get_functionspace(self.mesh2d, "CG", 1)
        self.bathymetry2d = Function(P1_2d, name='Bathymetry')
        self.bathymetry2d.assign(self.depth)
        self.horizontal_viscosity = Constant(0.5)
        self.horizontal_velocity_scale = Constant(1.5)
        self.flow_speed_ramped = Constant(0.0)

        # Turbine drag
        self.quadratic_drag_coefficient = Function(P1_2d, name='Cd')
        C_t = Constant(pi*self.thrust_coefficient/8)
        subset = self.mesh2d.cell_subset(self.array_ids[0])
        self.quadratic_drag_coefficient.interpolate(C_t, subset=subset)

        # Spatial discretisation
        self.element_family = 'dg-dg'

        # Temporal discretisation
        self.simulation_export_time = 2*60.0
        self.simulation_end_time = 8*3600.0
        self.timestepper_type = 'DIRK22'
        self.timestep = 60.0
        self.ramp_time = 1800.0

        # I/O
        self.fields_to_export = ['uv_2d']
        self.fields_to_export_hdf5 = []

    def apply_boundary_conditions(self, solver_obj):
        flux = Constant(self.depth*self.domain_width)*self.flow_speed_ramped
        inflow_tag = 1
        outflow_tag = 2
        inflow_bc = {'flux': -flux, 'elev': Constant(0.0)}
        outflow_bc = {'flux': flux, 'elev': Constant(0.0)}
        self.bnd_conditions = {
            inflow_tag: inflow_bc,
            outflow_tag: outflow_bc
        }
        solver_obj.bnd_functions['shallow_water'] = self.bnd_conditions

    def apply_initial_conditions(self, solver_obj):
        solver_obj.assign_initial_conditions(uv=Constant(as_vector([1.0e-04, 0])))

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
