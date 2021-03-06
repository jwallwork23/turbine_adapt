from pyroteus.thetis_compat import *


class FarmSolver(FlowSolver2d):
    """
    Modified solver which accepts :class:`ModelOptions2d` objects
    with more attributes than expected.
    """

    @unfrozen
    def __init__(self, options, mesh=None, keep_log=False):
        """
        :arg options: :class:`FarmOptions` parameter object
        :kwarg mesh: :class:`MeshGeometry` upon which to solve
        :kwarg bool keep_log: append to existing log file or create a new one?
        """
        self._initialized = False
        self.options = options
        self.mesh2d = mesh or options.mesh2d
        self.comm = self.mesh2d.comm

        self.dt = options.timestep
        self.simulation_time = 0
        self.iteration = 0
        self.i_export = 0
        self.next_export_t = self.simulation_time + options.simulation_export_time

        self.callbacks = thetis.callback.CallbackManager()
        self.fields = FieldDict()
        self._field_preproc_funcs = {}
        self.function_spaces = AttrDict()
        self.fields.bathymetry_2d = options.bathymetry2d

        self.export_initial_state = True
        self.sediment_model = None
        self.bnd_functions = {"shallow_water": {}, "tracer": {}, "sediment": {}}
        self.keep_log = keep_log
