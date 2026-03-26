"""
seiswave — Pure Python Surface Wave Seismogram Synthesis Library

A pure Python replacement for CPS (Computer Programs in Seismology)
modal summation workflow for computing synthetic surface wave seismograms
in a 1D layered earth model.

Modules:
    earth_model  — 1D layered earth model definitions
    propagator   — Thomson-Haskell / Dunkin dispersion solvers
    eigen        — Eigenfunction computation (displacement & stress)
    synth        — Modal summation → Green's functions → IFFT
    dispersion   — Phase-shift dispersion imaging (Park et al., 1999)
"""

from .earth_model import LayeredModel
from .propagator import find_dispersion
from .eigen import love_eigen, rayleigh_eigen
from .synth import compute_greens
from .dispersion import calculate_dispersion_image, plot_dispersion_image

__version__ = "0.1.0"
__all__ = [
    "LayeredModel",
    "find_dispersion",
    "love_eigen",
    "rayleigh_eigen",
    "compute_greens",
    "calculate_dispersion_image",
    "plot_dispersion_image",
]

