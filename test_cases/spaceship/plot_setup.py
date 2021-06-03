from turbine_adapt import *
from turbine_adapt.plotting import *
from options import SpaceshipOptions


# Set parameters
kwargs = {
    "interior_kw": {
        "linewidth": 0.1,
    },
    "boundary_kw": {
        "color": "k",
    },
}
font = {
    "family": "DejaVu Sans",
    "size": 18,
}
plt.rc("font", **font)
plt.rc("text", usetex=True)
patch_kwargs = {
    "facecolor": "none",
    "linewidth": 2,
}
options = SpaceshipOptions()
L = 1.05*options.domain_length
W = 1.05*options.domain_width

# Plot whole mesh
fig, axes = plt.subplots(figsize=(5, 4))
triplot(options.mesh2d, axes=axes, **kwargs)
axes.legend().remove()
axes.set_xlim([-L/2, L/2])
axes.set_ylim([-W/2, W/2])
axes.axis(False)
centres = [(6050, 0), (6450, 0)]
d = options.turbine_diameter
patches = []
for i, loc in enumerate(centres):
    patch_kwargs["edgecolor"] = f"C{2*i}"
    centre = (loc[0]-d/2, loc[1]-d/2)
    patch = ptch.Rectangle(centre, d, d, **patch_kwargs)
    axes.add_patch(patch)
    patches.append(patch)
plt.tight_layout()
di = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh", ext])))
axes.axis(True)

# Zoom of farm region
axes.set_xlim([-L/12, L/3])
axes.set_ylim([-W/6, W/6])
axes.set_xticks([])
axes.set_yticks([])
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_zoom", ext])))

# Zoom again
axes.set_xlim([5000, 7500])
axes.set_ylim([-1600, 2000])
axes.set_xticks([])
axes.set_yticks([])
plt.legend(patches, ['Turbine 1', 'Turbine 2'], ncol=2)
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["mesh_zoom_again", ext])))

# Plot bathymetry
fig, axes = plt.subplots(figsize=(5, 4))
eps = 1.0e-02
b = options.bathymetry2d
cbar_range = np.linspace(4.5 - eps, 25.5 + eps, 50)
cbar = fig.colorbar(tricontourf(b, axes=axes, levels=cbar_range, cmap='coolwarm_r'), ax=axes)
cbar.set_ticks(np.linspace(4.5, 25.5, 5))
cbar.set_label(r"Bathymetry $[m]$")
axes.axis(False)
plt.tight_layout()
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(di, ".".join(["bathymetry", ext])))

# Plot linear and exponential sponges
ticks = np.linspace(0.0, options.max_viscosity, 5)
ticks[0] = float(options.base_viscosity)
for sponge_type in ('linear', 'exponential'):
    fig, axes = plt.subplots(figsize=(5, 4))
    options.viscosity_sponge_type = sponge_type
    cbar_range = np.linspace(float(options.base_viscosity) - eps, options.max_viscosity + eps, 50)
    tc = tricontourf(options.horizontal_viscosity, levels=cbar_range, axes=axes, cmap='coolwarm')
    cbar = fig.colorbar(tc, ax=axes)
    cbar.set_ticks(ticks)
    cbar.set_label(r"Kinematic viscosity $[m^2\,s^{-1}]$")
    axes.axis(False)
    plt.tight_layout()
    for ext in ("jpg", "pdf"):
        plt.savefig(os.path.join(di, ".".join([sponge_type + "_sponge_viscosity", ext])))
