from turbine_adapt.parse import Parser
from options import ArrayOptions
from utils import *
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
parser.parse_setup()
parser.parse_approach()
parser.parse_metric_parameters()
parsed_args = parser.parse_args()
config = parsed_args.configuration
cols = np.sort([int(c) for c in parsed_args.columns])
ext = "".join([str(c) for c in cols])
if ext != "01234":
    parsed_args.approach += "_" + ext
approach = parsed_args.approach
assert approach not in ("fixed_mesh", "uniform_mesh")

# Load data
power = {}
time = {}
for a in ("uniform_mesh", approach):
    parsed_args["approach"] = a
    power[a], time[a] = get_data(config, ["run"], parsed_args)[:2]
assert np.allclose(time["uniform_mesh"], time[approach])
t = time["uniform_mesh"]
nt = len(t)

# Energy analysis
P = power["uniform_mesh"].reshape(nt, 5, 3)
E = time_integrate(np.sum(P, axis=2), t)
print(f"uniform_mesh   / {config:10s} energy            {np.round(E, 2)},  {np.round(np.sum(E), 2)}")
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
pwr = power["uniform_mesh"]
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
inflow = np.clip(np.outer(outputs[config]["P"], ones).reshape(1000, 15), tol, np.Inf)
power["uniform_mesh"] /= inflow
power[approach] /= inflow
P = power["uniform_mesh"].reshape(nt, 5, 3)
E = time_integrate(np.sum(P, axis=2), t)
print(f"uniform_mesh   / {config:10s} relative energy       {np.round(E, 2)}, {np.round(np.sum(E), 2)}")
head = f"{approach:15s}/ {config:10s}"
Ph = power[approach].reshape(nt, 5, 3)
Eh = time_integrate(np.sum(Ph, axis=2), t)
print(f"{head} relative energy       {np.round(Eh, 2)}, {np.round(np.sum(Eh), 2)}")
err = np.round(100 * np.abs((E - Eh) / E), 1)
overall = np.round(100 * np.abs((np.sum(E) - np.sum(Eh)) / np.sum(E)), 1)
print(f"{head} relative energy error {err} %, {overall} %")
