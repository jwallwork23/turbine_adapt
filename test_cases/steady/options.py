from turbine_adapt import *
from thetis.configuration import PositiveFloat, PositiveInteger


__all__ = ["SteadyTurbineOptions"]


class SteadyTurbineOptions(FarmOptions):

    # Turbine parameters
    turbine_diameter = PositiveFloat(18.0).tag(config=False)
    array_length = PositiveInteger(2).tag(config=False)
    array_width = PositiveInteger(1).tag(config=False)
    num_turbines = PositiveInteger(2).tag(config=False)

    # Domain specification
    domain_length = PositiveFloat(1200.0).tag(config=False)
    domain_width = PositiveFloat(500.0).tag(config=False)

    def __init__(self, mesh=None, **kwargs):
        super(SteadyTurbineOptions, self).__init__(**kwargs)
        self.array_ids = np.array([[2, 3]])
        self.farm_ids = (2, 3)

        self.mesh2d = mesh or Mesh('channel_0_1.msh')
        P1_2d = get_functionspace(self.mesh2d, "CG", 1)

        self.horizontal_viscosity = Constant(0.5)
        self.depth = 40.0
        self.bathymetry2d = Function(P1_2d)
        self.bathymetry2d.assign(self.depth)
        self.quadratic_drag_coefficient = Constant(0.0025)
        self.timestep = 20.0
        self.simulation_export_time = 20.0
        self.simulation_end_time = 18.0
        self.timestepper_type = 'SteadyState'
        self.timestepper_options.solver_parameters = {
            'mat_type': 'aij',
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_rtol': 1e-8,
            'snes_max_it': 100,
            'snes_monitor': None,
            'snes_converged_reason': None,
            'ksp_type': 'preonly',
            'ksp_converged_reason': None,
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
        }
        self.use_grad_div_viscosity_term = False
        self.use_lax_friedrichs_velocity = True
        self.lax_friedrichs_velocity_scaling_factor = Constant(1.0)
        self.use_grad_depth_viscosity_term = False
        self.element_family = 'dg-dg'

    def apply_boundary_conditions(self, solver_obj):
        solver_obj.bnd_functions['shallow_water'] = {
            1: {'uv': Constant(as_vector([5.0, 0.0]))},
            2: {'elev': Constant(0.0)},
            3: {'un': Constant(0.0)},
        }

    def apply_initial_conditions(self, solver_obj):
        solver_obj.assign_initial_conditions(uv=Constant(as_vector([5.0, 0.0])))
