import numpy as np
import pandas as pd
from utils import time_integrate


def ylim(y, config):
    """
    Determine if point y is directly upstream of a turbine
    in the first column.
    """
    if config == "aligned":
        return (-160 <= y <= -140) or (-10 <= y <= 10) or (140 <= y <= 160)
    else:
        return (-197.5 <= y <= -177.5) or (-47.5 <= y <= -27.5) or (102.5 <= y <= 122.5)


correct = True
W = 5
D = 20
H = 50
turbine_footprint = W * D
turbine_swept = np.pi * (D / 2) ** 2
turbine_cross_section = D * H
Ct = 2.985
CT = Ct
if correct:
    coeff = 0.5 * (1.0 + np.sqrt(1.0 - Ct * turbine_swept / turbine_cross_section))
    correction = 1 / coeff ** 2
    print(f"Speed coefficient: {coeff}")
    print(f"Thrust correction: {correction}")
    Ct *= correction
    print(f"Corrected thrust:  {Ct}")
else:
    coeff = 0.5 * (1.0 + np.sqrt(1.0 - Ct))
ct = 0.5 * Ct * turbine_swept / turbine_footprint
print(f"Turbine drag:      {ct}")
density = 1030.0
print(f"Coefficient:       {ct * density / 1.0e+06}")
print(f"Coefficient*area:  {ct * density / 1.0e+06 * 3 * turbine_swept}")


num_timesteps = 200  # exported timesteps
T_tide = 2232.0
t = np.linspace(T_tide, 1.5 * T_tide, num_timesteps) / 3600
dx = 1.0

outputs = {
    "aligned": {"xv": [], "P": []},
    "staggered": {"xv": [], "P": []},
}
for config in outputs:
    di = f"outputs/{config}/isotropic_dwr/target10000/Velocity2d/extracts"
    for i in range(num_timesteps):

        # Read velocity data for turbine region
        f = pd.read_csv(f"{di}/inflow_{i:06d}.csv", usecols=["x-velocity"])
        xv = np.array(f["x-velocity"])
        cnt = 0
        vbar = 0.0
        for y, v in zip(np.linspace(-500, 500, 1000), xv):
            if ylim(y, config):
                if np.isnan(v):
                    continue
                vbar += v
                cnt += 1
        if cnt > 0:
            vbar /= cnt
        # vbar = np.mean(np.abs(f["x-velocity"][200:999-200]))
        # vbar = np.mean(np.abs(f["x-velocity"]))

        # Compute mean inflow velocity
        outputs[config]["xv"].append(np.mean(xv))

        # Compute corresponding power output
        outputs[config]["P"].append(density * coeff ** 2 * ct * vbar ** 3 * turbine_footprint / 1.0e06)

    # Compute corresponding energy output
    outputs[config]["E"] = time_integrate(np.reshape(outputs[config]["P"], (num_timesteps, 1)), t)
    print(f"Energy output:      {outputs[config]['E']} MWh ({config})")
print(f"Staggered is {100*outputs['staggered']['E']/outputs['aligned']['E']-100}% higher")
print(f"Fluid speed is {100*np.mean((np.array(outputs['staggered']['xv'])/np.array(outputs['aligned']['xv'])))-100:.1f}% higher on average")
print(f"Power is {100*np.mean((np.array(outputs['staggered']['P'])/np.array(outputs['aligned']['P'])))-100:.1f}% higher on average")
