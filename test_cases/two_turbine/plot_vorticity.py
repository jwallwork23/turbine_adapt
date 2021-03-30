from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


# Loop over refinement levels
fig, axes = plt.subplots()
for level, colour in enumerate(['k', 'r', 'b', 'g', 'm']):

    # Load data
    input_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'level{:d}'.format(level))
    input_file = os.path.join(input_dir, 'diagnostic_vorticity.hdf5')
    if not os.path.exists(input_file):
        print("Cannot load data for refinement level {:d}.".format(level))
        continue
    with h5py.File(input_file, 'r') as f:
        zeta_max = np.array(f['max_vorticity'])
        zeta_min = np.array(f['min_vorticity'])
        time = np.array(f['time'])

    # Plot peak vorticities
    axes.plot(time, zeta_max, '-', label='Level {:d}'.format(level), color=colour)
    axes.plot(time, zeta_min, '--', color=colour)

# Save to file
axes.set_xlabel(r'Time [$\mathrm s$]')
axes.set_ylabel(r'Vorticity [$\mathrm s^{-1}$]')
axes.grid(True)
plt.tight_layout()
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
plt.savefig(os.path.join(plot_dir, 'peak_vorticities.pdf'))
