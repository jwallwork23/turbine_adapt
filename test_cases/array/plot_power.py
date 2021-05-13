from turbine_adapt.parse import Parser
from turbine_adapt.plotting import *
import h5py
import numpy as np
import os


# Parse arguments
parser = Parser()
parser.add_argument('-level', 0, help="""
    Mesh resolution level inside the refined region.
    Choose a value from [0, 1, 2, 3, 4] (default 0).""")
parser.add_argument('-approach', 'fixed_mesh')
parser.add_argument('-target', 5000*400)
args = parser.parse_args()
assert args.level in [0, 1, 2, 3, 4, 5]

# Load data
if args.approach == 'fixed_mesh':
    input_dir = 'level{:d}'.format(args.level)
    ext = '_level{:d}.pdf'.format(args.level)
else:
    input_dir = 'target{:.0f}'.format(args.target)
    ext = '_target{:.0f}.pdf'.format(args.target)
input_dir = os.path.join(os.path.dirname(__file__), 'outputs', args.approach, input_dir)
with h5py.File(os.path.join(input_dir, 'diagnostic_turbine.hdf5'), 'r') as f:
    power = np.array(f['current_power'])
    time = np.array(f['time'])
    time /= 4464.0

colours = ['black', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']

# Plot power output of each turbine
fig, axes = plt.subplots()
for i in range(15):
    _power = power[:, i]*1030.0/1.0e+06
    # _power = power[i, :]*1030.0/1.0e+06
    axes.plot(time, _power, label=f"Turbine {i+1}", color=colours[i//3])
axes.set_xlabel(r'Time/$T_{\mathrm{tide}}$')
axes.set_xlim([1, 1.5])
axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
axes.set_ylabel(r'Power output [$\mathrm{MW}$]')
axes.grid(True)
plt.tight_layout()
plot_dir = os.path.join(os.path.dirname(__file__), 'plots', args.approach)
plt.savefig(os.path.join(plot_dir, 'power_output' + ext))

# Plot power output of each column
fig, axes = plt.subplots()
for i in range(5):
    _power = power[:, 3*i] + power[:, 3*i+1] + power[:, 3*i+2]
    # _power = power[3*i, :] + power[3*i+1, :] + power[3*i+2, :]
    axes.plot(time, _power*1030.0/1.0e+06, label=f"{i+1}", color=colours[i])
axes.set_xlabel(r'Time/$T_{\mathrm{tide}}$')
axes.set_ylabel(r'Power output [$\mathrm{MW}$]')
axes.set_xlim([1, 1.5])
axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'power_output_column' + ext))

# Plot legend separately
fname = os.path.join(plot_dir, 'legend_column.pdf')
if not os.path.exists(fname):
    fig2, axes2 = plt.subplots()
    lines, labels = axes.get_legend_handles_labels()
    legend = axes2.legend(lines, labels, fontsize=18, frameon=False, ncol=5)
    fig2.canvas.draw()
    axes2.set_axis_off()
    bbox = legend.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    plt.savefig(fname, bbox_inches=bbox)

# Plot total power output
fig, axes = plt.subplots()
axes.plot(time, np.sum(power, axis=1)*1030.0/1.0e+06, label='Turbine {:d}'.format(i))
axes.set_xlim([1, 1.5])
axes.set_xticks([1, 1.125, 1.25, 1.375, 1.5])
axes.set_xlabel(r'Time [$\mathrm s$]')
axes.set_ylabel(r'Power output [$\mathrm{MW}$]')
axes.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, 'total_power_output' + ext))
