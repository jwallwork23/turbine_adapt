from utils import time_integrate
from options import ArrayOptions
import h5py
import numpy as np
import os
import sys


# Load data
approaches = ("fixed_mesh", "isotropic")
configs = ("aligned", "staggered")
power = {}
time = {}
for config in configs:
    power[config] = {}
    options = ArrayOptions(level=5, configuration=config, meshgen=True)
    end_time = options.ramp_time + options.simulation_end_time
    for approach in approaches:
        run = "level5" if approach == "fixed_mesh" else "target10000"
        output_dir = f"outputs/{config}/{approach}/{run}"
        fname = f"{output_dir}/diagnostic_turbine.hdf5"
        if not os.path.exists(fname):
            print(f"File {fname} does not exist")
            sys.exit(0)
        with h5py.File(fname, "r") as f:
            t = np.array(f["time"]) + options.tide_time
            P = np.array(f["current_power"])[t.flatten() <= end_time, :]
        power[config][approach] = P * 1030.0 / 1.0e06  # MW
        time[approach] = t[t <= end_time].flatten() / 3600.0  # h
    assert np.allclose(time["fixed_mesh"], time["isotropic"])
t = time["fixed_mesh"]
nt = len(t)

# Error analysis
p = 1
for config in configs:
    P = power[config]
    err = np.abs(P["fixed_mesh"] - P["isotropic"]).reshape(nt, 3, 5)
    Lp_err = time_integrate(np.sum(err**p, axis=0), t) ** (1 / p)
    pwr = power[config]["fixed_mesh"].reshape(nt, 3, 5)
    Lp_nrm = np.sqrt(time_integrate(np.sum(pwr**2, axis=0), t))
    rel_Lp_err = np.round(100 * Lp_err / Lp_nrm, 1)
    print(f"Relative L{p:.0f} error for {config:10s}: {rel_Lp_err} %,"
          f" {np.round(np.sum(rel_Lp_err), 1)} %")
