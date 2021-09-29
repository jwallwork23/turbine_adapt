from turbine_adapt import *
from turbine_adapt.plotting import *
from options import ArrayOptions
import sys
from matplotlib.lines import Line2D


# Parse for refinement level
parser = Parser(prog="test_cases/array/plot_mesh.py")
parser.add_argument('configuration', 'aligned', help="""
    Choose from 'aligned' and 'staggered'.
    """)
parsed_args = parser.parse_args()
config = parsed_args.configuration
assert config in ['aligned', 'staggered']

# Plotting
if COMM_WORLD.size > 1:
    print_output(f"Will not attempt to plot with {COMM_WORLD.size} processors."
                 " Run again in serial.")
    sys.exit(0)
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots', config))
kwargs = dict(interior_kw={"linewidth": 0.1}, boundary_kw={"color": "k"})
patch_kwargs = dict(facecolor="none", linewidth=2)
l = 15

# Set parameters
options = ArrayOptions(configuration=config)
L = options.domain_length
W = options.domain_width
w = options.turbine_width
d = options.turbine_diameter
deltax = 10.0*d
deltay = 7.5*d
centres = []
for i in range(-2, 3):
    for j in range(1, -2, -1):
        if config == "aligned":
            centres.append((i*deltax, j*deltay))
        elif config == "staggered":
            centres.append((i*deltax, (j + 0.25*(-1)**i)*deltay))
        else:
            raise NotImplementedError  # TODO

# Plot whole mesh
fig, axes = plt.subplots(figsize=(12, 4))
triplot(options.mesh2d, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2-l, L/2+l])
axes.set_ylim([-W/2-l, W/2+l])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
for i, loc in enumerate(centres):
    patch_kwargs["edgecolor"] = f"C{i // 3}"
    centre = (loc[0]-w/2, loc[1]-d/2)
    axes.add_patch(ptch.Rectangle(centre, w, d, **patch_kwargs))
plt.savefig(os.path.join(plot_dir, f"mesh_{config}.pdf"))

# Zoom in on array region
axes.set_xlim([-625, 625])
axes.set_ylim([-210, 210])
axes.set_xticks(np.linspace(-600, 600, 7))
axes.set_yticks(np.linspace(-200, 200, 9))
plt.savefig(os.path.join(di, f"mesh_zoom_{config}.pdf"))

colours = ['black', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']

# Plot domain without mesh
fig, axes = plt.subplots(figsize=(12, 5))
kwargs['interior_kw'] = {'color': 'w'}
kwargs['boundary_kw'].pop('color')
kwargs['boundary_kw']['colors'] = ['C2', 'C1', 'C2', 'C1']
kwargs['boundary_kw']['linewidths'] = 2.0
triplot(options.mesh2d, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2-l, L/2+l])
axes.set_ylim([-W/2-l, W/2+l])
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
patches = []
for i, loc in enumerate(centres):
    patch_kwargs["edgecolor"] = colours[i // 3]
    centre = (loc[0]-w/2, loc[1]-d/2)
    patch = ptch.Rectangle(centre, w, d, **patch_kwargs)
    if i % 3 == 0:
        patches.append(patch)
    axes.add_patch(patch)
w = 1700
d = 450
patch_kwargs["edgecolor"] = 'C0'
patch = ptch.Rectangle((-w/2, -d/2), w, d, **patch_kwargs)
patches.append(patch)
axes.add_patch(patch)
fname = f"domain_{config}"
plt.savefig(os.path.join(plot_dir, fname + '.pdf'))
plt.savefig(os.path.join(plot_dir, fname + '.jpg'))

plt.figure()
plt.gca().axis(False)
colors = ['C2', 'C1']
lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in colors]
for patch in patches:
    lines.append(patch)
labels = [r'$\Gamma_{\mathrm{freeslip}}$', r'$\Gamma_F$',
          'Column 1', ' Column 2', 'Column 3', 'Column 4', 'Column 5',
          'Zoom region']
plt.legend(lines, labels)
plt.tight_layout()
fname = "legend_domain"
plt.savefig(os.path.join(plot_dir, fname + '.pdf'))
plt.savefig(os.path.join(plot_dir, fname + '.jpg'))
