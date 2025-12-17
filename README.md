# Astrophysics Simulations

A comprehensive collection of physics and astrophysics simulations built from first principles. Each simulation is designed to be both **educational** and **visually stunning**.

## Quick Start

```bash
pip install -r requirements.txt

# Run any simulation
cd 01_orbital_mechanics && python solar_system.py
cd 01_orbital_mechanics && python hohmann_transfer.py
cd 02_nbody && python galaxy_collision.py
cd 03_stellar_physics && python hr_diagram.py
cd 04_black_holes && python raytracer.py
```

## Project Structure

```
astrophysics/
├── 01_orbital_mechanics/     # Kepler, orbits, space missions
│   ├── solar_system.py       # Interactive solar system [DONE]
│   └── hohmann_transfer.py   # Mission planning calculator
│
├── 02_nbody/                 # Gravitational N-body problems
│   ├── nbody_engine.py       # Core simulation engine
│   ├── galaxy_collision.py   # Milky Way vs Andromeda
│   └── lagrange_points.py    # L1-L5 stability visualization
│
├── 03_stellar_physics/       # Stars and stellar evolution
│   ├── hr_diagram.py         # Hertzsprung-Russell diagram
│   └── stellar_evolution.py  # Watch a star live and die
│
├── 04_black_holes/           # General relativity
│   ├── schwarzschild.py      # Geodesics around black holes
│   └── raytracer.py          # Gravitational lensing visualization
│
├── 05_cosmology/             # Universe-scale physics
│   └── expansion.py          # Friedmann equations, dark energy
│
└── shared/                   # Reusable physics code
    ├── constants.py          # Physical constants & planetary data
    ├── integrators.py        # Euler, Verlet, RK4 methods
    └── visualization.py      # Common plotting utilities
```

## Simulations

### 01 - Orbital Mechanics

| Simulation | Description | Physics |
|------------|-------------|---------|
| **Solar System** | Interactive N-body simulation with all planets | Newton's gravity, Verlet integration |
| **Hohmann Transfer** | Plan minimum-energy transfers between planets | Vis-viva equation, orbital mechanics |

### 02 - N-Body Problem

| Simulation | Description | Physics |
|------------|-------------|---------|
| **Galaxy Collision** | Watch the Milky Way and Andromeda merge | N-body dynamics, tidal forces |
| **Lagrange Points** | Visualize the 5 equilibrium points | Restricted 3-body problem |

### 03 - Stellar Physics

| Simulation | Description | Physics |
|------------|-------------|---------|
| **HR Diagram** | Plot real stars by temperature and luminosity | Blackbody radiation, stellar classification |
| **Stellar Evolution** | Input a star's mass, watch it evolve | Nuclear fusion, hydrostatic equilibrium |

### 04 - Black Holes & Relativity

| Simulation | Description | Physics |
|------------|-------------|---------|
| **Schwarzschild Geodesics** | Trace particle paths near black holes | General relativity, spacetime curvature |
| **Ray Tracer** | Visualize gravitational lensing | Null geodesics, photon spheres |

### 05 - Cosmology

| Simulation | Description | Physics |
|------------|-------------|---------|
| **Universe Expansion** | Model the expanding universe | Friedmann equations, dark energy |

## Physics Concepts

This project covers:

- **Classical Mechanics**: Newton's laws, orbital mechanics, Kepler's laws
- **Numerical Methods**: Euler, Velocity Verlet, Runge-Kutta integration
- **Thermodynamics**: Blackbody radiation, Stefan-Boltzmann law
- **Nuclear Physics**: Stellar fusion, CNO cycle, PP chain
- **General Relativity**: Schwarzschild metric, geodesic equations
- **Cosmology**: Friedmann equations, Hubble expansion, dark energy

## Controls (Common)

Most simulations share these controls:

| Key | Action |
|-----|--------|
| `SPACE` | Pause/Resume |
| `↑/↓` | Speed up/Slow down |
| `I/O` | Zoom in/out |
| `R` | Reset view |
| `Q/ESC` | Quit |

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

## Learning Resources

Each simulation includes comments explaining the physics. For deeper understanding:

- Orbital Mechanics: "Fundamentals of Astrodynamics" by Bate, Mueller, White
- N-Body: "Galactic Dynamics" by Binney & Tremaine
- Stellar Physics: "An Introduction to Modern Astrophysics" by Carroll & Ostlie
- General Relativity: "Spacetime and Geometry" by Sean Carroll
- Cosmology: "Introduction to Cosmology" by Barbara Ryden
