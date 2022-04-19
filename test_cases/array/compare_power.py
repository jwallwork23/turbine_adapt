from turbine_adapt.parse import Parser
from utils import time_integrate
from options import ArrayOptions
import h5py
import numpy as np
import os
import sys
from inflow_power import outputs


# Parse user input
parser = Parser("test_cases/array/compare_power.py")
parser.add_argument(
    "configuration",
    help="Name defining test case configuration",
    choices=["aligned", "staggered"],
)
parser.add_argument(
    "-c",
    "--columns",
    nargs="+",
    help="Turbine columns to use in QoI",
    default=[0, 1, 2, 3, 4],
)
parser.parse_approach()
parsed_args = parser.parse_args()
config = parsed_args.configuration
cols = np.sort([int(c) for c in parsed_args.columns])
ext = "".join([str(c) for c in cols])
if ext != "01234":
    parsed_args.approach += "_" + ext
approach = parsed_args.approach
assert approach != "fixed_mesh"

# Load data
power = {}
time = {}
for a in ("fixed_mesh", approach):
    options = ArrayOptions(level=5, configuration=config, meshgen=True)
    end_time = options.ramp_time + options.simulation_end_time
    run = "level5" if a == "fixed_mesh" else "target10000"
    output_dir = f"outputs/{config}/{a}/{run}"
    fname = f"{output_dir}/diagnostic_turbine.hdf5"
    if not os.path.exists(fname):
        print(f"File {fname} does not exist")
        sys.exit(0)
    with h5py.File(fname, "r") as f:
        t = np.array(f["time"]) + options.tide_time
        P = np.array(f["current_power"])[t.flatten() <= end_time, :]
    power[a] = P * 1030.0 / 1.0e06  # MW
    time[a] = t[t <= end_time].flatten() / 3600.0  # h
assert np.allclose(time["fixed_mesh"], time[approach])
t = time["fixed_mesh"]
nt = len(t)

# Energy analysis
P = power["fixed_mesh"].reshape(nt, 5, 3)
E = time_integrate(np.sum(P, axis=2), t)
print(f"fixed_mesh     / {config:10s} energy            {np.round(E, 2)},  {np.round(np.sum(E), 2)}")
head = f"{approach:15s}/ {config:10s}"
Ph = power[approach].reshape(nt, 5, 3)
Eh = time_integrate(np.sum(Ph, axis=2), t)
print(f"{head} energy            {np.round(Eh, 2)},  {np.round(np.sum(Eh), 2)}")
err = np.round(100 * np.abs((E - Eh) / E), 1)
overall = np.round(100 * np.abs((np.sum(E) - np.sum(Eh)) / np.sum(E)), 1)
print(f"{head} energy error      {err}%, {overall}%")


def Lp_norm(arr, p=1):
    return time_integrate(np.sum(np.abs(arr) ** p, axis=2), t) ** (1 / p)


# Error analysis
p = 1
pwr = power["fixed_mesh"]
head = f"{approach:15s}/ {config:10s}"
err = np.abs(pwr - power[approach])
rel_Lp_err = Lp_norm(err.reshape(nt, 5, 3)) / Lp_norm(pwr.reshape(nt, 5, 3))
rel_Lp_err = np.round(100 * rel_Lp_err, 1)
err = Lp_norm(err.reshape(nt, 1, 15)) / Lp_norm(pwr.reshape(nt, 1, 15))
err = np.round(100 * err, 1)
print(f"{head} relative L{p:.0f} error {rel_Lp_err}%, {err}%")


# Relative energy analysis
tol = 0.1
ones = np.ones((5, 3, 5))
power[approach] /= np.clip(np.outer(outputs[config]["P"], ones).reshape(1000, 15), tol, np.Inf)
P = power["fixed_mesh"].reshape(nt, 5, 3)
E = time_integrate(np.sum(P, axis=2), t)
print(f"fixed_mesh     / {config:10s} relative energy       {np.round(E, 2)}, {np.round(np.sum(E), 2)}")
head = f"{approach:15s}/ {config:10s}"
Ph = power[approach].reshape(nt, 5, 3)
Eh = time_integrate(np.sum(Ph, axis=2), t)
print(f"{head} relative energy       {np.round(Eh, 2)}, {np.round(np.sum(Eh), 2)}")
err = np.round(100 * np.abs((E - Eh) / E), 1)
overall = np.round(100 * np.abs((np.sum(E) - np.sum(Eh)) / np.sum(E)), 1)
print(f"{head} relative energy error {err} %, {overall} %")
