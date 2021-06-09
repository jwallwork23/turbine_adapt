from turbine_adapt import *


__all__ = ["PowerOutputCallback"]


class PowerOutputCallback(turbines.TurbineFunctionalCallback):
    """
    Subclass of :class:`TurbineFunctionalCallback` which reduces the verbosity of
    print statements.
    """
    def message_str(self, current_power, average_power, average_profit):
        power = 1030.0*sum(current_power)
        if power < 1.0e+03:
            return f"current power:     {power:5.3f} W"
        elif power < 1.0e+06:
            return f"current power:     {power/1.0e+03:5.3f} kW"
        elif power < 1.0e+09:
            return f"current power:     {power/1.0e+06:5.3f} MW"
        elif power < 1.0e+12:
            return f"current power:     {power/1.0e+09:5.3f} GW"
        else:
            return f"current power:     {power:10.4e} W"
