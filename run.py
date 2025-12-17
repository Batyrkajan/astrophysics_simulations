#!/usr/bin/env python
"""
Astrophysics Simulations Launcher
=================================

Run any simulation from the command line:
    python run.py                 # Show menu
    python run.py solar           # Run solar system
    python run.py blackhole       # Run black hole raytracer
    python run.py galaxy          # Run galaxy collision
"""

import sys
import os
import subprocess

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

SIMULATIONS = {
    'solar': {
        'name': 'Solar System',
        'path': '01_orbital_mechanics/solar_system.py',
        'desc': 'Interactive N-body solar system with all planets'
    },
    'hohmann': {
        'name': 'Hohmann Transfer',
        'path': '01_orbital_mechanics/hohmann_transfer.py',
        'desc': 'Plan minimum-energy transfers between planets'
    },
    'galaxy': {
        'name': 'Galaxy Collision',
        'path': '02_nbody/galaxy_collision.py',
        'desc': 'Watch two galaxies merge (like Milky Way + Andromeda)'
    },
    'lagrange': {
        'name': 'Lagrange Points',
        'path': '02_nbody/lagrange_points.py',
        'desc': 'Visualize L1-L5 equilibrium points'
    },
    'hr': {
        'name': 'HR Diagram',
        'path': '03_stellar_physics/hr_diagram.py',
        'desc': 'Interactive Hertzsprung-Russell diagram'
    },
    'stellar': {
        'name': 'Stellar Evolution',
        'path': '03_stellar_physics/stellar_evolution.py',
        'desc': 'Watch a star live and die'
    },
    'blackhole': {
        'name': 'Black Hole',
        'path': '04_black_holes/raytracer.py',
        'desc': 'Gravitational lensing ray tracer'
    },
    'cosmos': {
        'name': 'Universe Expansion',
        'path': '05_cosmology/expansion.py',
        'desc': 'Friedmann equations and dark energy'
    },
}


def print_menu():
    """Print available simulations."""
    print()
    print("=" * 60)
    print("  ASTROPHYSICS SIMULATIONS")
    print("=" * 60)
    print()
    print("  Available simulations:")
    print()

    for key, sim in SIMULATIONS.items():
        print(f"    {key:12} - {sim['name']}")
        print(f"                 {sim['desc']}")
        print()

    print("  Usage:")
    print("    python run.py <simulation>")
    print()
    print("  Example:")
    print("    python run.py solar")
    print("    python run.py blackhole")
    print()
    print("=" * 60)


def run_simulation(name):
    """Run a simulation by name."""
    if name not in SIMULATIONS:
        print(f"Unknown simulation: {name}")
        print(f"Available: {', '.join(SIMULATIONS.keys())}")
        return

    sim = SIMULATIONS[name]
    path = os.path.join(PROJECT_ROOT, sim['path'])

    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return

    print(f"\nLaunching {sim['name']}...")
    print(f"Path: {path}\n")

    # Run the simulation
    os.chdir(os.path.dirname(path))
    subprocess.run([sys.executable, os.path.basename(path)])


def main():
    if len(sys.argv) < 2:
        print_menu()

        # Interactive mode
        try:
            choice = input("Enter simulation name (or 'q' to quit): ").strip().lower()
            if choice and choice != 'q':
                run_simulation(choice)
        except (KeyboardInterrupt, EOFError):
            print("\n")
    else:
        sim_name = sys.argv[1].lower()
        if sim_name in ['-h', '--help', 'help']:
            print_menu()
        else:
            run_simulation(sim_name)


if __name__ == '__main__':
    main()
