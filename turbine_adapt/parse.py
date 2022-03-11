import argparse
import numpy as np


__all__ = ["Parser"]


def _check_positive(value, typ):
    tvalue = typ(value)
    if tvalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive {typ} value")
    return tvalue


positive_float = lambda value: _check_positive(value, float)
positive_int = lambda value: _check_positive(value, int)


def _check_nonnegative(value, typ):
    tvalue = typ(value)
    if tvalue < 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive {typ} value")
    return tvalue


nonnegative_float = lambda value: _check_nonnegative(value, float)
nonnegative_int = lambda value: _check_nonnegative(value, int)


def _check_in_range(value, typ, l, u):
    tvalue = typ(value)
    if not (tvalue >= l and tvalue <= u):
        raise argparse.ArgumentTypeError(f"{value} is not bounded by {(l, u)}")
    return tvalue


def bounded_float(l, u):
    def chk(value):
        return _check_in_range(value, float, l, u)

    return chk


class Parser(argparse.ArgumentParser):
    """
    Custom :class:`ArgumentParser` for `turbine_adapt`.
    """

    def __init__(self, prog):
        super().__init__(
            self, prog, formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

    def parse_setup(self):
        self.add_argument(
            "--level",
            help="Resolution level of initial mesh",
            type=nonnegative_int,
            default=0,
        )
        self.add_argument(
            "--ramp_level",
            help="Resolution level of spin-up run",
            type=nonnegative_int,
            default=5,
        )
        self.add_argument(
            "--end_time",
            help="Simulation end time in seconds",
            type=positive_float,
            default=None,
        )
        self.add_argument(
            "--num_tidal_cycles",
            help="Simulation end time in terms of tidal cycles",
            type=positive_float,
            default=0.5,
        )
        self.add_argument(
            "--num_meshes",
            help="Number of meshes in the fixed point iteration loop",
            type=positive_int,
            default=40,
        )

    def parse_convergence_criteria(self):
        self.add_argument(
            "--miniter",
            help="Minimum number of iterations",
            type=positive_int,
            default=3,
        )
        self.add_argument(
            "--maxiter",
            help="Maximum number of iterations",
            type=positive_int,
            default=5,
        )
        self.add_argument(
            "--qoi_rtol",
            help="Relative tolerance for QoI",
            type=positive_float,
            default=0.005,
        )
        self.add_argument(
            "--element_rtol",
            help="Element count tolerance",
            type=positive_float,
            default=0.005,
        )

    def parse_approach(self):
        self.add_argument(
            "-a",
            "--approach",
            help="Adaptive approach to consider",
            choices=["fixed_mesh", "isotropic_dwr", "anisotropic_dwr"],
            default="isotropic_dwr",
        )

    def parse_indicator(self):
        self.add_argument(
            "-i",
            "--indicator",
            help="Error indicator formulation",
            choices=["difference_quotient"],
            default="difference_quotient",
        )
        self.add_argument(
            "--flux_form",
            help="Toggle whether to use the flux form of the difference quotient indicator",
            action="store_true",
        )

    def parse_metric_parameters(self):
        self.add_argument(
            "--target_complexity",
            help="Target metric complexity",
            type=positive_float,
            default=10000.0,
        )
        self.add_argument(
            "--norm_order",
            help="Order p for L^p normalisation",
            type=bounded_float(1.0, np.inf),
            default=1.0,
        )
        self.add_argument(
            "--h_min",
            help="Minimum metric magnitude",
            type=positive_float,
            default=0.01,
        )
        self.add_argument(
            "--h_max",
            help="Maximum metric magnitude",
            type=positive_float,
            default=100.0,
        )
        self.add_argument(
            "--turbine_h_max",
            help="Maximum metric magnitude inside turbine footprint",
            type=positive_float,
            default=2.0,
        )

    def parse_plotting(self):
        self.add_argument(
            "--plot_bathymetry",
            help="Toggle plotting of bathymetry field",
            action="store_true",
        )
        self.add_argument(
            "--plot_drag",
            help="Toggle plotting of drag field",
            action="store_true",
        )
        self.add_argument(
            "--plot_metric",
            help="Toggle plotting of metric",
            action="store_true",
        )

    def parse_loading(self):
        self.add_argument(
            "--load_metric",
            help="Toggle loading of metric data from file",
            action="store_true",
        )
        self.add_argument(
            "--load_index",
            help="Index for loading mesh and metric data from file",
            type=positive_int,
            default=0,
        )
