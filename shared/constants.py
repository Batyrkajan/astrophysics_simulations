"""
Physical constants for astrophysics simulations.
All values in SI units unless otherwise noted.
"""

# Fundamental constants
G = 6.67430e-11          # Gravitational constant [m^3 kg^-1 s^-2]
c = 2.99792458e8         # Speed of light [m/s]
h = 6.62607015e-34       # Planck constant [J s]
k_B = 1.380649e-23       # Boltzmann constant [J/K]
sigma = 5.670374419e-8   # Stefan-Boltzmann constant [W m^-2 K^-4]

# Distance units
AU = 1.495978707e11      # Astronomical Unit [m]
LY = 9.4607e15           # Light year [m]
PC = 3.0857e16           # Parsec [m]

# Time units
DAY = 86400              # Seconds in a day
YEAR = 365.25 * DAY      # Seconds in a Julian year

# Solar properties
M_SUN = 1.98892e30       # Solar mass [kg]
R_SUN = 6.9634e8         # Solar radius [m]
L_SUN = 3.828e26         # Solar luminosity [W]
T_SUN = 5778             # Solar effective temperature [K]

# Earth properties
M_EARTH = 5.97237e24     # Earth mass [kg]
R_EARTH = 6.371e6        # Earth radius [m]

# Planetary data (mass in kg, semi-major axis in AU, orbital velocity in m/s)
PLANETS = {
    'Sun':     {'mass': 1.989e30, 'distance': 0,          'velocity': 0,     'color': 'yellow'},
    'Mercury': {'mass': 3.285e23, 'distance': 0.387 * AU, 'velocity': 47360, 'color': 'gray'},
    'Venus':   {'mass': 4.867e24, 'distance': 0.723 * AU, 'velocity': 35020, 'color': 'orange'},
    'Earth':   {'mass': 5.972e24, 'distance': 1.000 * AU, 'velocity': 29780, 'color': 'blue'},
    'Mars':    {'mass': 6.390e23, 'distance': 1.524 * AU, 'velocity': 24070, 'color': 'red'},
    'Jupiter': {'mass': 1.898e27, 'distance': 5.203 * AU, 'velocity': 13070, 'color': 'orange'},
    'Saturn':  {'mass': 5.683e26, 'distance': 9.537 * AU, 'velocity': 9690,  'color': 'gold'},
    'Uranus':  {'mass': 8.681e25, 'distance': 19.19 * AU, 'velocity': 6810,  'color': 'lightblue'},
    'Neptune': {'mass': 1.024e26, 'distance': 30.07 * AU, 'velocity': 5430,  'color': 'blue'},
}
