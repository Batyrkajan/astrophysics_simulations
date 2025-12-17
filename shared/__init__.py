"""
Shared utilities for astrophysics simulations.

Modules:
- constants: Physical constants and planetary data
- integrators: Numerical integration methods (Euler, Verlet, RK4)
- visualization: Common plotting and animation utilities
"""

from .constants import *
from .integrators import Simulation, euler_step, verlet_step, rk4_step
from .visualization import (
    temperature_to_rgb, velocity_to_color, create_colormap,
    setup_dark_figure, add_info_panel,
    draw_orbit, draw_body, draw_vector, add_starfield,
    create_trail, format_time, format_distance, format_mass
)
