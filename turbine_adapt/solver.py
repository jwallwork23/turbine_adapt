from turbine_adapt import *


class FarmSolver(solver2d.FlowSolver2d):
    """
    Modified solver which accepts :class:`ModelOptions2d` objects
    with more attributes than expected.
    """
    def __init__(self, options):
        self._initialized = False
        self.options = options
        self.mesh2d = options.mesh2d
        self.comm = self.mesh2d.comm

        self.dt = options.timestep
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + options.simulation_export_time

        self.callbacks = callback.CallbackManager()
        self.fields = FieldDict()
        self.function_spaces = AttrDict()
        self.fields.bathymetry_2d = options.bathymetry2d

        self.export_initial_state = True
        self.sediment_model = None
        self.bnd_functions = {'shallow_water': {}, 'tracer': {}, 'sediment': {}}
        self._isfrozen = True
