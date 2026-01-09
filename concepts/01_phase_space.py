"""
Phase Space Visualization
=========================

Understanding what numerical integration REALLY does.

Key insight: We're not approximating a trajectory.
We're creating a DISCRETE DYNAMICAL SYSTEM that (hopefully)
preserves the properties of the continuous one.
"""

import numpy as np
import matplotlib.pyplot as plt

# Simple Harmonic Oscillator: d²x/dt² = -x
# In phase space: dx/dt = v, dv/dt = -x
# This is the simplest non-trivial dynamical system.

def plot_phase_space_flow():
    """
    Visualize the vector field in phase space.

    At every point (x, v), the system tells us:
    - dx/dt = v      (move right if v > 0)
    - dv/dt = -x     (accelerate left if x > 0)

    The arrows show WHERE THE SYSTEM WANTS TO GO at each point.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create grid of points in phase space
    x = np.linspace(-2, 2, 20)
    v = np.linspace(-2, 2, 20)
    X, V = np.meshgrid(x, v)

    # The vector field: (dx/dt, dv/dt) = (v, -x)
    dX = V           # dx/dt = v
    dV = -X          # dv/dt = -x (harmonic oscillator)

    # Normalize arrows for visualization
    magnitude = np.sqrt(dX**2 + dV**2)
    dX_norm = dX / magnitude
    dV_norm = dV / magnitude

    # Plot vector field
    ax.quiver(X, V, dX_norm, dV_norm, magnitude, cmap='coolwarm', alpha=0.7)

    # The TRUE solution is a circle: x² + v² = constant (energy conservation)
    theta = np.linspace(0, 2*np.pi, 100)
    for r in [0.5, 1.0, 1.5]:
        ax.plot(r*np.cos(theta), r*np.sin(theta), 'k--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Position (x)', fontsize=12)
    ax.set_ylabel('Velocity (v)', fontsize=12)
    ax.set_title('Phase Space Vector Field\n"Where does the system want to go?"', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)

    # Add annotation
    ax.annotate('Circles = constant energy\n(true solutions)',
                xy=(1.2, 1.2), fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig('concepts/output/01a_phase_space_flow.png', dpi=150)
    plt.close()
    print("\nThe arrows show the FLOW - where phase space points move.")
    print("True solutions trace circles (energy conservation).")
    print("Question: Do our numerical methods preserve this?")


def compare_integrators():
    """
    The critical comparison: What do different integrators do to phase space?

    EULER: Spirals outward (gains energy) - BREAKS physics
    VERLET: Stays on (approximate) circle - PRESERVES physics
    RK4: Very accurate short-term, but still drifts long-term
    """

    def euler_step(x, v, dt):
        """
        Euler: Use current derivatives to step forward.

        x_new = x + v * dt
        v_new = v + a * dt = v + (-x) * dt

        Problem: We use OLD x to compute acceleration,
        but particle has already moved!
        """
        a = -x  # acceleration at current position
        x_new = x + v * dt
        v_new = v + a * dt
        return x_new, v_new

    def verlet_step(x, v, dt):
        """
        Velocity Verlet: The key insight is SYMMETRY.

        1. Half-step velocity using current acceleration
        2. Full-step position using half-stepped velocity
        3. Compute NEW acceleration at new position
        4. Complete velocity step using average acceleration

        This is SYMPLECTIC - it preserves phase space volume.
        """
        a = -x  # current acceleration

        # Half velocity step
        v_half = v + 0.5 * a * dt

        # Full position step
        x_new = x + v_half * dt

        # New acceleration
        a_new = -x_new

        # Complete velocity step
        v_new = v_half + 0.5 * a_new * dt

        return x_new, v_new

    def rk4_step(x, v, dt):
        """
        RK4: Sample the vector field at 4 points, take weighted average.

        Very accurate per step, but NOT symplectic.
        For long simulations, Verlet beats RK4 despite lower order!
        """
        def derivs(x, v):
            return v, -x  # dx/dt, dv/dt

        k1_x, k1_v = derivs(x, v)
        k2_x, k2_v = derivs(x + 0.5*dt*k1_x, v + 0.5*dt*k1_v)
        k3_x, k3_v = derivs(x + 0.5*dt*k2_x, v + 0.5*dt*k2_v)
        k4_x, k4_v = derivs(x + dt*k3_x, v + dt*k3_v)

        x_new = x + (dt/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        v_new = v + (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        return x_new, v_new

    # Initial conditions
    x0, v0 = 1.0, 0.0  # Start at x=1, v=0
    dt = 0.1
    n_steps = 500  # Many orbits to see long-term behavior

    # Run all three
    results = {}
    for name, stepper in [('Euler', euler_step),
                          ('Verlet', verlet_step),
                          ('RK4', rk4_step)]:
        x, v = x0, v0
        trajectory = [(x, v)]
        energies = [0.5 * (x**2 + v**2)]  # E = 0.5*(x² + v²) for SHO

        for _ in range(n_steps):
            x, v = stepper(x, v, dt)
            trajectory.append((x, v))
            energies.append(0.5 * (x**2 + v**2))

        results[name] = {
            'trajectory': np.array(trajectory),
            'energies': np.array(energies)
        }

    # Plot phase space trajectories
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = {'Euler': '#e74c3c', 'Verlet': '#2ecc71', 'RK4': '#3498db'}

    for ax, name in zip(axes, ['Euler', 'Verlet', 'RK4']):
        traj = results[name]['trajectory']

        # True solution circle
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5,
                linewidth=2, label='True (circle)')

        # Numerical trajectory
        ax.plot(traj[:, 0], traj[:, 1], color=colors[name],
                linewidth=0.5, alpha=0.8)
        ax.plot(traj[0, 0], traj[0, 1], 'ko', markersize=8, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 's', color=colors[name],
                markersize=8, label='End')

        ax.set_xlabel('Position (x)')
        ax.set_ylabel('Velocity (v)')
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concepts/output/01b_integrator_comparison.png', dpi=150)
    plt.close()

    # Plot energy over time
    fig, ax = plt.subplots(figsize=(12, 5))

    time = np.arange(n_steps + 1) * dt

    for name in ['Euler', 'Verlet', 'RK4']:
        energies = results[name]['energies']
        # Normalize to show relative error
        relative_energy = (energies - energies[0]) / energies[0] * 100
        ax.plot(time, relative_energy, color=colors[name],
                linewidth=2, label=name)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy Error (%)', fontsize=12)
    ax.set_title('Energy Conservation: The Critical Test', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('concepts/output/01c_energy_conservation.png', dpi=150)
    plt.close()

    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    for name in ['Euler', 'Verlet', 'RK4']:
        final_energy_error = (results[name]['energies'][-1] - 0.5) / 0.5 * 100
        print(f"{name:8s}: Final energy error = {final_energy_error:+.2f}%")
    print("="*60)
    print("\nEuler GAINS energy (spiral out) - physically wrong!")
    print("Verlet OSCILLATES around true energy - bounded error!")
    print("RK4 slowly DRIFTS - accurate short-term, problematic long-term")


if __name__ == '__main__':
    import os
    os.makedirs('concepts/output', exist_ok=True)

    print("="*60)
    print("  PHASE SPACE AND NUMERICAL INTEGRATION")
    print("="*60)
    print("\nThis script demonstrates WHY different integrators behave")
    print("differently, and why Verlet is preferred for physics.\n")

    plot_phase_space_flow()
    print("\n" + "-"*60 + "\n")
    compare_integrators()
