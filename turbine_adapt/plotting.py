import matplotlib
import matplotlib.patches as ptch  # noqa
import matplotlib.pyplot as plt  # noqa
from mpltools import annotation  # noqa
import os  # noqa


matplotlib.rc("text", usetex=True)
matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 24

whiteline = matplotlib.lines.Line2D([0], [0], color="w")
