from turbine_adapt import *


__all__ = ["PowerOutputCallback"]


class PowerOutputCallback(turbines.TurbineFunctionalCallback):
    """
    Subclass of :class:`TurbineFunctionalCallback` which reduces the verbosity of
    print statements.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = np.shape(self.solver_obj.options.array_ids)

    def message_str(self, current_power, average_power, average_profit):
        power = np.reshape(1030.0 * current_power, self.shape)
        overall = np.sum(power.flatten())
        unit = "W"
        if overall < 1.0e3:
            pass
        elif overall < 1.0e06:
            power /= 1.0e03
            unit = "kW"
        elif overall < 1.0e09:
            power /= 1.0e06
            unit = "MW"
        elif overall < 1.0e12:
            power /= 1.0e09
            unit = "GW"
        overall = np.sum(power.flatten())
        colwise = ", ".join([f"{col:8.4f}" for col in np.sum(power, axis=0)])
        unit = f"({unit})"
        return f"power {unit:4s}: {overall:8.4f} [{colwise}]"
