from options import ArrayOptions
import h5py
import numpy as np
import os
import sys


def time_integrate(arr, times):
    """
    Time integrate an array of turbine power outputs
    that were obtained using Crank-Nicolson.

    :arg arr: the (n_timesteps, n_turbines) array
    :arg times: the corresponding array of times
    """
    times = times.flatten()
    dt = times[1] - times[0]
    zeros = np.zeros(arr.shape[1:])
    off1 = np.vstack((zeros, arr))
    off2 = np.vstack((arr, zeros))
    return dt * 0.5 * np.sum(off1 + off2, axis=0)


def get_data(config, modes, namespace, eps=1.0e-05):
    """
    :arg config: configuration, from 'aligned' and 'staggered'
    :arg modes: a list of simulation modes, from 'ramp' and 'run'
    :arg namespace: a :class:`NameSpace` of user input
    """
    level = namespace.level
    approach = namespace.approach
    end_time = namespace.end_time

    mode = modes[0] if len(modes) == 1 else "both"
    options = ArrayOptions(level=level, configuration=config, meshgen=True)
    if end_time is None:
        end_time = options.ramp_time
        if mode != "ramp":
            end_time += namespace.num_tidal_cycles * options.tide_time

    # Load data
    if approach == "fixed_mesh":
        run = f"level{level}"
    else:
        run = f"target{namespace.target_complexity:.0f}"
    output_dir = f"outputs/{config}/{approach}/{run}"
    power = np.array([]).reshape((0, 15))
    time = np.array([]).reshape((0, 1))
    for m in modes:
        ramp = m == "ramp"
        input_dir = output_dir + "/ramp" if ramp else output_dir
        fname = f"{input_dir}/diagnostic_turbine.hdf5"
        if not os.path.exists(fname):
            print(f"File {fname} does not exist")
            sys.exit(0)
        with h5py.File(fname, "r") as f:
            t = np.array(f["time"])
            if not ramp:
                t += options.tide_time
            time = np.vstack((time, t))
            power = np.vstack((power, np.array(f["current_power"])))
    power = power[time.flatten() <= end_time, :] * 1030.0 / 1.0e06  # MW
    time = time[time <= end_time].flatten()

    # Sort and remove duplicates
    sort = np.argsort(time)
    time = time[sort]
    power = power[sort, :]
    keep = [not np.isclose(t, time[i+1]) for i, t in enumerate(time[:-1])]
    keep.append(True)
    keep = np.array(keep)
    power = power[keep, :]
    time = time[keep]

    # Calculate energy per halfcycle
    p = power.copy()
    t = time.copy()
    dt = 0.5 * options.tide_time
    energy = []
    energy_time = []
    while len(t) > 0:
        tnext = t[0] + dt
        energy.append(sum(time_integrate(p[t <= tnext, :], t[t <= tnext])))
        energy_time.append(t[0] + 0.2 * dt)
        p = p[t >= tnext]
        t = t[t >= tnext]

    # Convert to cycle time
    end_time /= options.tide_time
    time /= options.tide_time
    energy_time = np.array(energy_time) / options.tide_time
    energy = np.array(energy) / 3600.0  # MWh

    # Plot formatting
    ticks = []
    if mode != "run":
        ticks += [0, 0.25, 0.5, 0.75, 1]
    else:
        ticks += list(np.arange(1, end_time + eps, 0.125))
    # NOTE: Assumes ramp is just one tidal cycle
    if mode == "both":
        ticks += list(np.arange(1, end_time + eps, 0.25))
    figsize = (4.4 + 2 * (end_time - 1 if mode == "run" else end_time), 4.8)

    return power, time, energy, energy_time, end_time, run, ticks, figsize
