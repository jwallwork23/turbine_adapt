from turbine_adapt import *


__all__ = ["PeakVorticityCallback", "PowerOutputCallback"]


class PeakVorticityCallback(DiagnosticCallback):
    """
    Computes the peak vorticities of the horizontal velocity field
    using an L2 projection.
    """
    name = 'vorticity'
    variable_names = ['min_vorticity', 'max_vorticity']

    def __init__(self, *args, plot=True, **kwargs):
        super(PeakVorticityCallback, self).__init__(*args, **kwargs)
        self._outfile = None
        if plot:
            outputdir = create_directory(os.path.join(self.outputdir, 'Vorticity2d'))
            self._outfile = File(os.path.join(outputdir, 'Vorticity2d.pvd'))
            self._simulation_export_time = self.solver_obj.options.simulation_export_time
            self._next_export_time = self.solver_obj.simulation_time + self._simulation_export_time

    def _initialize(self):
        self._initialized = True
        if hasattr(self, '_outfile') and self._outfile is not None:
            self._outfile._topology = None
        uv = self.solver_obj.fields.uv_2d
        uv_perp = as_vector([-uv[1], uv[0]])
        mesh = self.solver_obj.mesh2d
        x, y = SpatialCoordinate(mesh)

        # Set up a system to recover zeta by L2 projection
        P1_2d = self.solver_obj.function_spaces.P1_2d
        self._zeta = Function(P1_2d, name='Vorticity')
        test, trial = TestFunction(P1_2d), TrialFunction(P1_2d)
        n = FacetNormal(mesh)
        a = inner(test, trial)*dx
        L = inner(test, dot(uv_perp, n))*ds - inner(grad(test), uv_perp)*dx
        problem = LinearVariationalProblem(a, L, self._zeta)
        params = {
            'ksp_type': 'gmres',
            'ksp_gmres_restart': 20,
            'ksp_rtol': 1.0e-05,
            'pc_type': 'sor',
        }
        self.solver = LinearVariationalSolver(problem, solver_parameters=params)

    @property
    def zeta(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self._initialize()
        self.solver.solve()
        return self._zeta

    def __call__(self):
        zeta = self.zeta
        if self._outfile is not None:
            if self.solver_obj.simulation_time >= self._next_export_time - 1.0e-05:
                self._next_export_time += self._simulation_export_time
                self._outfile.write(zeta)
        zeta = zeta.vector().gather()
        return (zeta.min(), zeta.max())

    def message_str(self, *args):
        line = 'min/max vorticity: {:14.8e}, {:14.8e}'
        return line.format(args[0], args[1])


class PowerOutputCallback(turbines.TurbineFunctionalCallback):
    """
    Subclass of :class:`TurbineFunctionalCallback` which reduces the verbosity of
    print statements.
    """
    def message_str(self, current_power, average_power, average_profit):
        power = sum(current_power)
        if power < 1.0e+03:
            return 'current power:     {:5.3f} W'.format(power)
        elif power < 1.0e+06:
            return 'current power:     {:5.3f} kW'.format(power/1.0e+03)
        elif power < 1.0e+09:
            return 'current power:     {:5.3f} MW'.format(power/1.0e+06)
        elif power < 1.0e+12:
            return 'current power:     {:5.3f} GW'.format(power/1.0e+09)
        else:
            return 'current power:     {:10.4e} W'.format(power)
