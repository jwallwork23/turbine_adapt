from __future__ import absolute_import
from turbine_adapt import *
from turbine_adapt.error_estimation import ErrorEstimator
from options import SteadyTurbineOptions


# Set parameters
options = SteadyTurbineOptions()
options.create_tidal_farm()

# Solve forward
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
options.apply_initial_conditions(solver_obj)
solver_obj.iterate()

# Solve adjoint
ts = solver_obj.timestepper
V = solver_obj.function_spaces.V_2d
dFdu = derivative(ts.F, ts.solution, TrialFunction(V))
dFdu_transpose = adjoint(dFdu)
adj_sol = Function(V)
ee = ErrorEstimator(options, norm_type='L2')
uv, elev = split(ts.solution)
J = ee.drag_coefficient*dot(uv, uv)**1.5*dx
print("Power output = {:.4e}".format(assemble(J)))
dJdu = derivative(J, ts.solution, TestFunction(V))
solve(dFdu_transpose == dJdu, adj_sol, solver_parameters=ts.solver_parameters)

# Estimate error
uv, elev = ts.solution.split()
z, zeta = adj_sol.split()
dq = ee.difference_quotient(uv, elev, z, zeta)
File('outputs/difference_quotient.pvd').write(dq)
