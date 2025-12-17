"""
Solar System Simulation
=======================

A physically accurate N-body simulation of our solar system.

Features:
- Velocity Verlet integration (symplectic, energy-conserving)
- Vectorized NumPy computations for performance
- Real planetary data from NASA
- Energy conservation monitoring
- Interactive controls

Controls:
    SPACE  - Pause/Resume
    UP/DOWN - Speed up/Slow down
    I/O    - Zoom in/out
    1-4    - View presets (inner, outer, full, jupiter)
    R      - Reset view
    Q/ESC  - Quit

Run:
    python solar_system.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

from shared.constants import G, AU, DAY, PLANETS


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

class NBodySystem:
    """
    N-body gravitational system with Velocity Verlet integration.

    Uses vectorized NumPy operations for performance.
    Tracks energy to verify numerical accuracy.
    """

    def __init__(self, masses, positions, velocities, softening=0.0):
        """
        Initialize the N-body system.

        Args:
            masses: (n,) array of masses in kg
            positions: (n, 2) array of positions in m
            velocities: (n, 2) array of velocities in m/s
            softening: softening length to prevent singularities (default 0)
        """
        self.masses = np.array(masses, dtype=np.float64)
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.softening = softening
        self.n_bodies = len(masses)

        # Pre-compute accelerations for Verlet
        self.accelerations = self._compute_accelerations()

        # Track initial energy for conservation check
        self.initial_energy = self.total_energy()

    def _compute_accelerations(self):
        """
        Compute gravitational accelerations using vectorized operations.

        Uses broadcasting to compute all pairwise interactions efficiently.
        O(n²) complexity but vectorized for speed.
        """
        # Position differences: r_ij = r_j - r_i
        # Shape: (n, n, 2) where [i, j] = position of j relative to i
        dr = self.positions[np.newaxis, :, :] - self.positions[:, np.newaxis, :]

        # Distances: |r_ij|
        # Shape: (n, n)
        r_squared = np.sum(dr**2, axis=2) + self.softening**2
        r = np.sqrt(r_squared)

        # Avoid division by zero on diagonal
        np.fill_diagonal(r, 1.0)

        # Gravitational acceleration: a_i = sum_j (G * m_j * r_ij / |r_ij|³)
        # Shape: (n, n, 2)
        acc_components = G * self.masses[np.newaxis, :, np.newaxis] * dr / r[:, :, np.newaxis]**3

        # Zero out self-interaction
        for i in range(self.n_bodies):
            acc_components[i, i] = 0.0

        # Sum over all other bodies
        accelerations = np.sum(acc_components, axis=1)

        return accelerations

    def step(self, dt):
        """
        Advance system by one timestep using Velocity Verlet.

        Velocity Verlet is symplectic, meaning it conserves phase space volume
        and exhibits excellent long-term energy conservation.
        """
        # Position update: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        self.positions += self.velocities * dt + 0.5 * self.accelerations * dt**2

        # Compute new accelerations at new positions
        new_accelerations = self._compute_accelerations()

        # Velocity update: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        self.velocities += 0.5 * (self.accelerations + new_accelerations) * dt

        # Store for next step
        self.accelerations = new_accelerations

    def kinetic_energy(self):
        """Total kinetic energy: KE = Σ 0.5 * m * v²"""
        return 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)

    def potential_energy(self):
        """Total gravitational potential energy: PE = -Σᵢ<ⱼ G*mᵢ*mⱼ/rᵢⱼ"""
        PE = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r = np.linalg.norm(self.positions[i] - self.positions[j])
                PE -= G * self.masses[i] * self.masses[j] / r
        return PE

    def total_energy(self):
        """Total mechanical energy (should be conserved)."""
        return self.kinetic_energy() + self.potential_energy()

    def energy_error(self):
        """Relative energy error from initial state."""
        if self.initial_energy == 0:
            return 0.0
        return abs((self.total_energy() - self.initial_energy) / self.initial_energy)

    def center_of_mass(self):
        """Compute barycenter of the system."""
        total_mass = np.sum(self.masses)
        return np.sum(self.masses[:, np.newaxis] * self.positions, axis=0) / total_mass


# =============================================================================
# VISUALIZATION
# =============================================================================

class SolarSystemVisualization:
    """
    Interactive visualization of the solar system simulation.
    """

    # View presets: (xlim, ylim) in AU
    VIEWS = {
        'inner': 2.0,      # Mercury to Mars
        'outer': 35.0,     # Include Neptune
        'full': 6.5,       # Default view to Jupiter
        'jupiter': 8.0,    # Focus on Jupiter system
    }

    # Planet display sizes (not to scale, for visibility)
    PLANET_SIZES = {
        'Sun': 20, 'Mercury': 4, 'Venus': 6, 'Earth': 6, 'Mars': 5,
        'Jupiter': 14, 'Saturn': 12, 'Uranus': 9, 'Neptune': 9
    }

    def __init__(self, system, names, colors, dt):
        self.system = system
        self.names = names
        self.colors = colors
        self.dt = dt
        self.time = 0.0

        # Animation state
        self.paused = False
        self.speed_multiplier = 1
        self.current_view = 'full'
        self.view_scale = self.VIEWS['full']

        # History for trails
        self.max_trail = 500
        self.history = [self.system.positions.copy()]

        self._setup_figure()
        self._setup_controls()

    def _setup_figure(self):
        """Create the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')

        # Main simulation axes
        self.ax = self.fig.add_axes([0.05, 0.1, 0.7, 0.85])
        self.ax.set_facecolor('#0a0a15')
        self.ax.set_aspect('equal')
        self._set_view(self.view_scale)

        # Style axes
        self.ax.tick_params(colors='#444444', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333333')
        self.ax.set_xlabel('Distance (AU)', color='#666666', fontsize=9)
        self.ax.set_ylabel('Distance (AU)', color='#666666', fontsize=9)

        # Add starfield background
        self._add_starfield()

        # Create planet markers and trails
        self.dots = []
        self.trails = []
        self.labels = []

        for i, name in enumerate(self.names):
            size = self.PLANET_SIZES.get(name, 6)

            # Planet dot with glow effect
            dot, = self.ax.plot([], [], 'o', color=self.colors[i],
                               markersize=size, zorder=10)

            # Orbital trail
            trail, = self.ax.plot([], [], '-', color=self.colors[i],
                                 alpha=0.3, linewidth=1, zorder=5)

            # Label
            label = self.ax.text(0, 0, '', color=self.colors[i], fontsize=8,
                                ha='left', va='bottom', zorder=15,
                                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

            self.dots.append(dot)
            self.trails.append(trail)
            self.labels.append(label)

        # Info panel (right side)
        self.info_ax = self.fig.add_axes([0.78, 0.1, 0.2, 0.85])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.title = self.ax.set_title('Solar System Simulation',
                                        color='white', fontsize=14, fontweight='bold')

        # Time display
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      color='white', fontsize=11,
                                      verticalalignment='top', family='monospace')

        # Controls help
        self.help_text = self.ax.text(0.02, 0.02,
                                      'SPACE:pause  ↑↓:speed  I/O:zoom  1-4:views  R:reset',
                                      transform=self.ax.transAxes, color='#555555',
                                      fontsize=8, verticalalignment='bottom')

        # Energy display
        self.energy_text = self.ax.text(0.98, 0.98, '', transform=self.ax.transAxes,
                                        color='#44ff44', fontsize=9, ha='right',
                                        verticalalignment='top', family='monospace')

    def _add_starfield(self, n_stars=200):
        """Add random background stars."""
        np.random.seed(42)  # Reproducible starfield
        limit = 40 * AU
        x = np.random.uniform(-limit, limit, n_stars)
        y = np.random.uniform(-limit, limit, n_stars)
        sizes = np.random.uniform(0.1, 1.0, n_stars)
        alphas = np.random.uniform(0.3, 0.8, n_stars)

        for i in range(n_stars):
            self.ax.plot(x[i], y[i], '.', color='white',
                        markersize=sizes[i], alpha=alphas[i], zorder=1)

    def _set_view(self, scale_au):
        """Set the view limits in AU."""
        self.view_scale = scale_au
        limit = scale_au * AU
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)

        # Update tick labels to show AU
        ticks = np.linspace(-scale_au, scale_au, 5)
        self.ax.set_xticks(ticks * AU)
        self.ax.set_yticks(ticks * AU)
        self.ax.set_xticklabels([f'{t:.0f}' for t in ticks])
        self.ax.set_yticklabels([f'{t:.0f}' for t in ticks])

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'up':
            self.speed_multiplier = min(self.speed_multiplier * 2, 64)
        elif event.key == 'down':
            self.speed_multiplier = max(self.speed_multiplier // 2, 1)
        elif event.key in ['i', 'I']:
            self._set_view(max(1.0, self.view_scale * 0.7))
        elif event.key in ['o', 'O']:
            self._set_view(min(50.0, self.view_scale * 1.4))
        elif event.key == '1':
            self._set_view(self.VIEWS['inner'])
        elif event.key == '2':
            self._set_view(self.VIEWS['full'])
        elif event.key == '3':
            self._set_view(self.VIEWS['jupiter'])
        elif event.key == '4':
            self._set_view(self.VIEWS['outer'])
        elif event.key in ['r', 'R']:
            self._set_view(self.VIEWS['full'])
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def _update_info_panel(self):
        """Update the information panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.info_ax.text(0.5, 0.98, 'PLANET DATA', transform=self.info_ax.transAxes,
                         color='white', fontsize=11, fontweight='bold', ha='center', va='top')

        y = 0.92
        for i, name in enumerate(self.names):
            if name == 'Sun':
                continue

            pos = self.system.positions[i]
            vel = self.system.velocities[i]

            # Distance from Sun in AU
            dist = np.linalg.norm(pos) / AU

            # Orbital velocity in km/s
            speed = np.linalg.norm(vel) / 1000

            # Display
            self.info_ax.text(0.05, y, f'● {name}', transform=self.info_ax.transAxes,
                             color=self.colors[i], fontsize=9, fontweight='bold')
            self.info_ax.text(0.05, y - 0.035, f'  {dist:.2f} AU | {speed:.1f} km/s',
                             transform=self.info_ax.transAxes, color='#888888', fontsize=8)
            y -= 0.09

        # Simulation stats
        self.info_ax.text(0.5, 0.08, '─' * 15, transform=self.info_ax.transAxes,
                         color='#333333', ha='center', fontsize=8)
        self.info_ax.text(0.05, 0.04, f'Speed: {self.speed_multiplier}x',
                         transform=self.info_ax.transAxes, color='#666666', fontsize=8)

    def animate(self, frame):
        """Animation update function."""
        if not self.paused:
            # Run physics steps
            for _ in range(self.speed_multiplier):
                self.system.step(self.dt)
                self.time += self.dt

            # Record history for trails
            self.history.append(self.system.positions.copy())
            if len(self.history) > self.max_trail:
                self.history.pop(0)

        # Convert history to array for plotting
        history_arr = np.array(self.history)

        # Update plot elements
        for i in range(self.system.n_bodies):
            x, y = self.system.positions[i]

            # Update dot
            self.dots[i].set_data([x], [y])

            # Update trail
            self.trails[i].set_data(history_arr[:, i, 0], history_arr[:, i, 1])

            # Update label (offset based on view scale)
            if self.names[i] != 'Sun':
                offset = 0.05 * self.view_scale * AU
                self.labels[i].set_position((x + offset, y + offset))
                self.labels[i].set_text(self.names[i])

        # Update time display
        years = self.time / DAY / 365.25
        status = "PAUSED" if self.paused else "RUNNING"
        self.time_text.set_text(f'Time: {years:.2f} years  [{status}]')

        # Update energy conservation
        energy_err = self.system.energy_error()
        color = '#44ff44' if energy_err < 1e-6 else '#ffff44' if energy_err < 1e-4 else '#ff4444'
        self.energy_text.set_text(f'ΔE/E: {energy_err:.2e}')
        self.energy_text.set_color(color)

        # Update info panel every few frames
        if frame % 10 == 0:
            self._update_info_panel()

        return self.dots + self.trails + self.labels + [self.time_text, self.energy_text]

    def run(self):
        """Start the animation."""
        self.anim = FuncAnimation(self.fig, self.animate, frames=None,
                                  interval=30, blit=False, cache_frame_data=False)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the solar system simulation."""

    # Configure which bodies to simulate
    # Using all planets for full solar system
    body_names = ['Sun', 'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

    n_bodies = len(body_names)

    # Build arrays from planetary data
    masses = np.array([PLANETS[name]['mass'] for name in body_names])
    colors = [PLANETS[name]['color'] for name in body_names]

    # Initial positions: all bodies on x-axis at their orbital distance
    positions = np.zeros((n_bodies, 2))
    for i, name in enumerate(body_names):
        positions[i, 0] = PLANETS[name]['distance']

    # Initial velocities: all moving in +y direction (circular orbit approximation)
    velocities = np.zeros((n_bodies, 2))
    for i, name in enumerate(body_names):
        velocities[i, 1] = PLANETS[name]['velocity']

    # Create physics system
    # Small softening prevents numerical issues if bodies get very close
    system = NBodySystem(masses, positions, velocities, softening=1e8)

    # Timestep: 0.5 days gives good accuracy for inner planets
    dt = 0.5 * DAY

    print("=" * 50)
    print("  SOLAR SYSTEM SIMULATION")
    print("=" * 50)
    print(f"  Bodies: {', '.join(body_names)}")
    print(f"  Timestep: {dt/DAY:.1f} days")
    print(f"  Integrator: Velocity Verlet (symplectic)")
    print()
    print("  Controls:")
    print("    SPACE   - Pause/Resume")
    print("    ↑/↓     - Speed up/Slow down")
    print("    I/O     - Zoom in/out")
    print("    1-4     - View presets")
    print("    Q/ESC   - Quit")
    print("=" * 50)

    # Create visualization and run
    viz = SolarSystemVisualization(system, body_names, colors, dt)
    viz.run()


if __name__ == '__main__':
    main()
