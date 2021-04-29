__all__ = ["Parser"]


class Parser(object):
    """
    Custom argument parser with pre-defined arguments which
    allows setting defaults.
    """
    def __init__(self, **kwargs):
        import argparse
        self._parser = argparse.ArgumentParser(**kwargs)
        self._help = {
            'approach': {'type': str, 'msg': """
                Mesh adaptation approach (default '{:s}').
                """},
            'level': {'type': int, 'msg': """
                Resolution level of initial mesh (default {:d}).
                """},
            'end_time': {'type': float, 'msg': """
                Simulation end time in seconds (default {:.1f}).
                """},
            'num_tidal_cycles': {'type': float, 'msg': """
                Simulation end time in terms of tidal cycles (default {:.1f}).
                """},
            'num_meshes': {'type': int, 'msg': """
                Number of meshes in the fixed point iteration loop (default {:d}).
                """},
            'miniter': {'type': int, 'msg': """
                Minimum number of fixed point iterations (default {:d}).
                """},
            'maxiter': {'type': int, 'msg': """
                Maximum number of fixed point iterations (default {:d}). If set to zero,
                mesh adaptation is not applied.
                """},
            'element_rtol': {'type': float, 'msg': """
                Relative tolerance for element count convergence (default {:.4e})
                """},
            'qoi_rtol': {'type': float, 'msg': """
                Relative tolerance for quantity of interest convergence (default {:.4e})
                """},
            'norm_order': {'type': float, 'msg': """
                Order p used in L-p space-time normalisation (default {:}). Choose a value
                greater than or equal to one, or 'inf' to specify L-infinity normalisation.
                """},
            'target': {'type': float, 'msg': """
                Target *spatial* complexity (default {:.4e}), i.e. metric complexity
                associated with a single mesh iteration.
                """},
            'h_min': {'type': float, 'msg': """
                Minimum tolerated element size in metres (default {:.4e}.
                """},
            'h_max': {'type': float, 'msg': """
                Maximum tolerated element size in metres (default {:.4e}.
                """},
            'plot_bathymetry': {'type': bool, 'msg': """
                Toggle plotting of bathymetry field (default {:b}).
                """},
            'plot_drag': {'type': bool, 'msg': """
                Toggle plotting of drag field (default {:b}).
                """},
            'plot_metric': {'type': bool, 'msg': """
                Toggle plotting of metric field (default {:b}).
                """},
            'load_metric': {'type': bool, 'msg': """
                Toggle loading metric data from file (default {:b}).
                """},
            'adjoint_projection': {'type': bool, 'msg': """
                Toggle whether to project adjoint solutions using conservative
                interpolation operator or its adjoint (default {:b}).
                """},
            'flux_form': {'type': bool, 'msg': """
                Toggle whether to use the flux form of the difference quotient
                error indicator (default {:b}).
                """},
        }
        self._added = {}

    def add_argument(self, label, default, help=None):
        tag = label[1:] if label[0] == '-' else label
        if help is not None:
            kwargs = dict(help=help)
        elif tag in self._help:
            kwargs = dict(help=self._help[tag]['msg'].format(default))
        else:
            kwargs = {}
        if tag not in self._help:
            self._help[tag] = dict(type=type(default))
        self._parser.add_argument(label, **kwargs)
        self._added[tag] = default

    def parse_args(self):
        from thetis.utility import AttrDict
        parsed = self._parser.parse_args()
        out = AttrDict()
        for tag in self._added:
            p = getattr(parsed, tag)
            if tag == 'norm_order' and p == 'inf':
                out[tag] = 'inf'
            elif self._help[tag]['type'] == bool and self._added[tag]:
                out[tag] = False if tag == '0' else True
            else:
                out[tag] = self._help[tag]['type'](p or self._added[tag])
        return out
