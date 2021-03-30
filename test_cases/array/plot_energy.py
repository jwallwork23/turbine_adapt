from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


# Parse arguments
parser = Parser()
parser.add_argument('-approach', 'fixed_mesh')
args = parser.parse_args()

# Loop over runs
fig, axes = plt.subplots()
elements = [29860, 45334, 52162, 120548, 160222]  # TODO: avoid hard-code
E = []
x = []
targets = [1250*400, 2500*400, 5000*400, 10000*400]
for level, target in enumerate(targets):

    # Load data
    if args.approach == 'fixed_mesh':
        input_dir = 'level{:d}'.format(level)
    else:
        input_dir = 'target{:.0f}'.format(target)
    input_dir = os.path.join(os.path.dirname(__file__), 'outputs', args.approach, input_dir)
    input_file = os.path.join(input_dir, 'diagnostic_turbine.hdf5')
    if not os.path.exists(input_file):
        print("Cannot load data for refinement level {:d}.".format(level))
        continue
    with h5py.File(input_file, 'r') as f:
        power = np.array(f['current_power'])
        time = np.array(f['time'])
    assert len(power) == len(time)
    power = power[len(power)//2:]
    time = time[len(time)//2:]
    total_power = np.sum(power, axis=1)*1030.0/1.0e+06/3600

    # Compute energy output using trapezium rule on each timestep
    energy = 0
    for i in range(len(total_power)-1):
        energy += 0.5*(time[i+1] - time[i])*(total_power[i+1] + total_power[i])
    E.append(energy)
    x.append(3*elements[level] if args.approach == 'fixed_mesh' else targets[level])

# Plot
fig, axes = plt.subplots(figsize=(6, 4))
if args.approach == 'fixed_mesh':
    axes.semilogx(x, E, '--x')
else:
    axes.plot(x, E, '--x')
axes.set_xlabel(r'DoF count' if args.approach == 'fixed_mesh' else 'Target complexity')
axes.set_ylabel(r'Energy [$\mathrm{MW\,h}$]')
axes.grid(True)
plt.tight_layout()
plot_dir = os.path.join(os.path.dirname(__file__), 'plots', args.approach)
plt.savefig(os.path.join(plot_dir, 'energy_output.pdf'))
