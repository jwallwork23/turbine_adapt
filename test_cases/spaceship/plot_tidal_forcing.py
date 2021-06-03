from thetis import *
import matplotlib.pyplot as plt
from options import SpaceshipOptions


# Set parameters
font = {
    "family": "DejaVu Sans",
    "size": 20,
}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plotting_kwargs = {
    "annotation_clip": False,
    "color": "b",
    "arrowprops": {
        "color": "b",
        "arrowstyle": "<->",
    },
}
options = SpaceshipOptions()

# Interpolate forcing onto time range
interp = options.tidal_forcing_interpolator
time_seconds = np.linspace(0.0, options.tidal_forcing_end_time, 1001)
forcing = interp(time_seconds)
time_seconds -= options.ramp_time
time_hours = time_seconds/3600
time_days = time_hours/24

# Plot the spin-up period only
fig, axes = plt.subplots(figsize=(8, 6))
axes.plot(time_hours, forcing, color='grey')
axes.axhline(0, linestyle=':', color='lightgrey')
axes.set_xlabel(r"Time $[h]$")
axes.set_ylabel(r"Tidal forcing $[m]$")
axes.set_xlim([-options.ramp_time/3600, 0])
axes.set_ylim([-4, 4])
axes.set_xticks([-18, -12, -6, 0])
plot_dir = create_directory(os.path.join(os.path.dirname(__file__), 'plots'))
plt.tight_layout()
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["spin_up", ext])))

# Plot the forcing over the whole time range
fig, axes = plt.subplots(figsize=(18, 6))
axes.plot(time_days, forcing, color='grey')
axes.set_xlabel("Time [days]")
axes.set_ylabel("Tidal forcing [m]")

# Annotate the spin-up period
axes.set_xticks(range(17))
r = options.ramp_time/3600/24
axes.axvline(0, linestyle='--', color='b')
axes.axhline(0, linestyle=':', color='lightgrey')
axes.set_xlim([-r, 15])
axes.set_ylim([-4, 4])
axes.annotate("", xy=(-r, -5), xytext=(0, -5), **plotting_kwargs)
axes.annotate("Spin-up period", xy=(-1.3, -5.6), xytext=(-1.3, -5.6), color="b", annotation_clip=False)

# Add second x-axis with non-dimensionalised time
non_dimensionalise = lambda time: 24*3600*time/options.tide_time
dimensionalise = lambda time: 24*3600*time*options.tide_time
secax = axes.secondary_xaxis('top', functions=(non_dimensionalise, dimensionalise))
secax.set_xlabel("Time/Tidal period")
secax.set_xticks(range(32))

# Save plot
plt.tight_layout()
for ext in ("jpg", "pdf"):
    plt.savefig(os.path.join(plot_dir, ".".join(["tidal_forcing", ext])))
