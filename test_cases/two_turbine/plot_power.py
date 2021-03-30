from turbine_adapt.plotting import *
import argparse
import h5py
import numpy as np
import os


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-level', help="""
    Mesh resolution level inside the refined region
    Choose a value from [0, 1, 2, 3, 4] (default 1)""")
args = parser.parse_args()
level = int(args.level or 1)
assert level in [0, 1, 2, 3, 4]

# Load data
input_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'level{:d}'.format(level))
with h5py.File(os.path.join(input_dir, 'diagnostic_turbine.hdf5'), 'r') as f:
    power = np.array(f['current_power'])
    time = np.array(f['time'])

# Plot power output
fig, axes = plt.subplots()
axes.plot(time, power[:, 0]*1030.0/1.0e+06, label='First turbine')
axes.plot(time, power[:, 1]*1030.0/1.0e+06, label='Second turbine')
axes.set_xlabel(r'Time [$\mathrm s$]')
axes.set_ylabel(r'Power [$\mathrm{MW}$]')
axes.set_yticks([0, 5, 10, 15, 20])
axes.grid(True)
plt.tight_layout()
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
plt.savefig(os.path.join(plot_dir, 'power_output_level{:d}.pdf'.format(level)))
