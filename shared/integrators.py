"""
Numerical integration methods for physics simulations.

Available integrators:
- Euler: Simple, fast, but energy drifts over time
- Velocity Verlet: Symplectic, excellent energy conservation
- RK4: High accuracy, good for non-conservative systems
"""

import numpy as np


def euler_step(positions, velocities, acceleration_func, dt):
    """
    Simple Euler integration.

    Fast but energy will drift over long simulations.
    Use for quick prototyping, not production.

    Args:
        positions: (n_bodies, dims) array of positions
        velocities: (n_bodies, dims) array of velocities
        acceleration_func: function(positions) -> accelerations
        dt: timestep

    Returns:
        new_positions, new_velocities
    """
    accelerations = acceleration_func(positions)
    new_velocities = velocities + accelerations * dt
    new_positions = positions + velocities * dt
    return new_positions, new_velocities


def verlet_step(positions, velocities, acceleration_func, dt, prev_accelerations=None):
    """
    Velocity Verlet integration (leapfrog variant).

    Symplectic integrator - conserves energy over long timescales.
    Best choice for orbital mechanics and N-body simulations.

    Args:
        positions: (n_bodies, dims) array of positions
        velocities: (n_bodies, dims) array of velocities
        acceleration_func: function(positions) -> accelerations
        dt: timestep
        prev_accelerations: accelerations from previous step (optional)

    Returns:
        new_positions, new_velocities, new_accelerations
    """
    if prev_accelerations is None:
        prev_accelerations = acceleration_func(positions)

    # Update positions using current velocity and acceleration
    new_positions = positions + velocities * dt + 0.5 * prev_accelerations * dt**2

    # Calculate new accelerations at new positions
    new_accelerations = acceleration_func(new_positions)

    # Update velocities using average of old and new accelerations
    new_velocities = velocities + 0.5 * (prev_accelerations + new_accelerations) * dt

    return new_positions, new_velocities, new_accelerations


def rk4_step(positions, velocities, acceleration_func, dt):
    """
    4th-order Runge-Kutta integration.

    High accuracy per step. Good for systems with non-conservative
    forces or when you need precise trajectories.

    Args:
        positions: (n_bodies, dims) array of positions
        velocities: (n_bodies, dims) array of velocities
        acceleration_func: function(positions) -> accelerations
        dt: timestep

    Returns:
        new_positions, new_velocities
    """
    # k1
    k1_v = acceleration_func(positions)
    k1_x = velocities

    # k2
    k2_v = acceleration_func(positions + 0.5 * dt * k1_x)
    k2_x = velocities + 0.5 * dt * k1_v

    # k3
    k3_v = acceleration_func(positions + 0.5 * dt * k2_x)
    k3_x = velocities + 0.5 * dt * k2_v

    # k4
    k4_v = acceleration_func(positions + dt * k3_x)
    k4_x = velocities + dt * k3_v

    # Combine
    new_positions = positions + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
    new_velocities = velocities + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

    return new_positions, new_velocities


class Simulation:
    """
    Generic physics simulation runner.

    Example usage:
        sim = Simulation(positions, velocities, accel_func, dt)
        sim.set_integrator('verlet')
        history = sim.run(n_steps=1000)
    """

    INTEGRATORS = {
        'euler': 'euler',
        'verlet': 'verlet',
        'rk4': 'rk4'
    }

    def __init__(self, positions, velocities, acceleration_func, dt):
        self.positions = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.acceleration_func = acceleration_func
        self.dt = dt
        self.integrator = 'verlet'  # Default to best energy conservation
        self._prev_accelerations = None

    def set_integrator(self, name):
        """Set integration method: 'euler', 'verlet', or 'rk4'"""
        if name not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {name}. Choose from {list(self.INTEGRATORS.keys())}")
        self.integrator = name
        self._prev_accelerations = None

    def step(self):
        """Advance simulation by one timestep."""
        if self.integrator == 'euler':
            self.positions, self.velocities = euler_step(
                self.positions, self.velocities, self.acceleration_func, self.dt
            )
        elif self.integrator == 'verlet':
            self.positions, self.velocities, self._prev_accelerations = verlet_step(
                self.positions, self.velocities, self.acceleration_func,
                self.dt, self._prev_accelerations
            )
        elif self.integrator == 'rk4':
            self.positions, self.velocities = rk4_step(
                self.positions, self.velocities, self.acceleration_func, self.dt
            )

    def run(self, n_steps, save_every=1):
        """
        Run simulation for n_steps.

        Args:
            n_steps: number of timesteps
            save_every: save state every N steps (for memory efficiency)

        Returns:
            history: (n_saved, n_bodies, dims) array of positions
        """
        n_saved = n_steps // save_every
        history = np.zeros((n_saved, *self.positions.shape))

        save_idx = 0
        for step in range(n_steps):
            if step % save_every == 0:
                history[save_idx] = self.positions.copy()
                save_idx += 1
            self.step()

        return history

    def total_energy(self, masses):
        """Calculate total mechanical energy (kinetic + potential)."""
        # Kinetic energy
        KE = 0.5 * np.sum(masses[:, np.newaxis] * self.velocities**2)

        # Potential energy
        from .constants import G
        PE = 0
        n = len(masses)
        for i in range(n):
            for j in range(i + 1, n):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                PE -= G * masses[i] * masses[j] / r

        return KE + PE
