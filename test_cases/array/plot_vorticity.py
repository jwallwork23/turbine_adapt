from turbine_adapt.argparse import Parser
from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


# Parse arguments
parser = Parser()
parser.add_argument('-approach', 'fixed_mesh')
args = parser.parse_args()

# Setup iterables
colours = ['b', 'r', 'k', 'g', 'm']
levels = range(5)
targets = [1250*400, 2500*400, 5000*400, 10000*400]
if args.approach == 'fixed_mesh':
    dofs = ['89,580', '136,002', '156,486', '361,644', '480,666']
    elements = ['29,860', '45,334', '52,162', '120,548', '160,222']
    labels = ['{:s} elements\n{:s} DoFs'.format(elements[level], dofs[level]) for level in levels]
else:
    labels = [str(target) for target in targets]

# Loop over refinement levels
fig, axes = plt.subplots(figsize=(8, 5))
width = 0.25
for level, (colour, label) in enumerate(zip(colours, labels)):

    # Load data
    if args.approach == 'fixed_mesh':
        input_dir = 'level{:d}'.format(level)
    else:
        input_dir = 'target{:.0f}'.format(targets[level])
    input_dir = os.path.join(os.path.dirname(__file__), 'outputs', args.approach, input_dir)
    input_file = os.path.join(input_dir, 'diagnostic_vorticity.hdf5')
    if not os.path.exists(input_file):
        print("Cannot load data for refinement level {:d}.".format(level))
        continue
    with h5py.File(input_file, 'r') as f:
        zeta_max = np.array(f['max_vorticity'])
        zeta_min = np.array(f['min_vorticity'])
        time = np.array(f['time'])

    # Plot peak vorticities
    axes.plot(time, zeta_max, '-', label=label, color=colour, linewidth=width)
    axes.plot(time, zeta_min, '--', color=colour, linewidth=width)

# Save to file
axes.set_xlabel(r'Time [$\mathrm s$]')
axes.set_ylabel(r'Vorticity [$\mathrm s^{-1}$]')
axes.grid(True)
plt.tight_layout()
plot_dir = os.path.join(os.path.dirname(__file__), 'plots', args.approach)
# plt.savefig(os.path.join(plot_dir, 'peak_vorticities.pdf'))
plt.savefig(os.path.join(plot_dir, 'peak_vorticities.jpg'))

# Plot legend separately
fig2, axes2 = plt.subplots()
lines, labels = axes.get_legend_handles_labels()
legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=1)
fig2.canvas.draw()
axes2.set_axis_off()
bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
# plt.savefig(os.path.join(plot_dir, 'legend_vorticity.pdf'), bbox_inches=bbox)
plt.savefig(os.path.join(plot_dir, 'legend_vorticity.jpg'), bbox_inches=bbox)

# Plot relative vorticity
fig, axes = plt.subplots(figsize=(8, 5))
for level, (colour, label) in enumerate(zip(colours, labels)):

    # Load data
    if args.approach == 'fixed_mesh':
        input_dir = 'level{:d}'.format(level)
    else:
        input_dir = 'target{:.0f}'.format(targets[level])
    input_dir = os.path.join(os.path.dirname(__file__), 'outputs', args.approach, input_dir)
    input_file = os.path.join(input_dir, 'diagnostic_vorticity.hdf5')
    if not os.path.exists(input_file):
        print("Cannot load data for refinement level {:d}.".format(level))
        continue
    with h5py.File(input_file, 'r') as f:
        zeta_max = np.array(f['max_vorticity'])
        zeta_min = np.array(f['min_vorticity'])
        time = np.array(f['time'])

    # Plot peak vorticities
    axes.plot(time, zeta_max/zeta_max.max(), '-', label=label, color=colour, linewidth=width)
    axes.plot(time, zeta_min/(-zeta_min.min()), '--', color=colour, linewidth=width)

# Save to file
axes.set_xlabel(r'Time [$\mathrm s$]')
axes.set_ylabel(r'Vorticity [$\mathrm s^{-1}$]')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'relative_peak_vorticities.pdf'))
