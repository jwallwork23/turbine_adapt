import numpy as np


def time_integrate(arr, times):
    """
    Time integrate an array of turbine power outputs
    that were obtained using Crank-Nicolson.

    :arg arr: the (n_timesteps, n_turbines) array
    :arg times: the corresponding array of times
    """
    times = times.flatten()
    dt = times[1] - times[0]
    zeros = np.zeros(arr.shape[1])
    off1 = np.vstack((zeros, arr))
    off2 = np.vstack((arr, zeros))
    return dt * 0.5 * np.sum(off1 + off2, axis=0)
