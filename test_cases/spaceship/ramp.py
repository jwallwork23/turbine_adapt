from turbine import *
from options import SpaceshipOptions


# Set parameters
options = ArrayOptions()
options.simulation_end_time = options.ramp_time
options.create_tidal_farm()
output_dir = os.path.join(options.output_directory, 'fixed_mesh')
options.output_directory = create_directory(output_dir)

# Solve
solver_obj = FarmSolver(options)
options.apply_boundary_conditions(solver_obj)
options.apply_initial_conditions(solver_obj)
solver_obj.iterate(update_forcings=options.update_forcings, export_func=options.export_func)

# Store to checkpoint
uv, elev = solver_obj.fields.solution_2d.split()
fname = "ramp"
with DumbCheckpoint(os.path.join(output_dir, fname), mode=FILE_CREATE) as chk:
    chk.store(uv)
    chk.store(elev)
