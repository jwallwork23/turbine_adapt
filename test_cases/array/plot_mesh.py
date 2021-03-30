from turbine_adapt import *
from turbine_adapt.plotting import *
from options import ArrayOptions
import sys


# Set parameters
options = ArrayOptions()

# Plotting
if COMM_WORLD.size > 1:
    msg = "Will not attempt to plot with {:d} processors. Run again in serial."
    print_output(msg.format(COMM_WORLD.size))
    sys.exit(0)
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
kwargs = dict(interior_kw={"linewidth": 0.1}, boundary_kw={"color": "k"})
patch_kwargs = dict(facecolor="none", linewidth=2)
L = options.domain_length
W = options.domain_width
l = 15

# Plot whole mesh
fig, axes = plt.subplots(figsize=(12, 6))
triplot(options.mesh2d, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2-l, L/2+l])
axes.set_ylim([-W/2-l, W/2+l])
axes.set_xlabel(r"$x$-coordinate $[\mathrm m]$")
axes.set_ylabel(r"$y$-coordinate $[\mathrm m]$")
axes.set_yticks(np.linspace(-W/2, W/2, 5))
plt.tight_layout()
for i, loc in enumerate(options.turbine_geometries):
    patch_kwargs["edgecolor"] = "C{:d}".format(i // 3)
    centre = (loc[0]-loc[2]/2, loc[1]-loc[3]/2)
    axes.add_patch(ptch.Rectangle(centre, loc[2], loc[3], **patch_kwargs))
plt.savefig(os.path.join(di, 'mesh.pdf'))

# Zoom in on array region
axes.set_xlim([-625, 625])
axes.set_ylim([-210, 210])
axes.set_xticks(np.linspace(-600, 600, 7))
axes.set_yticks(np.linspace(-200, 200, 9))
plt.savefig(os.path.join(di, "mesh_zoom.pdf"))
