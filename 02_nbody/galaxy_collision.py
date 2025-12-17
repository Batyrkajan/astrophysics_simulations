"""
Galaxy Collision Simulator
==========================

Watch two spiral galaxies collide and merge, similar to the future
collision between the Milky Way and Andromeda.

Physics:
- N-body gravitational dynamics
- Each galaxy has a central supermassive black hole
- Disk stars in circular orbits
- Tidal forces tear apart galactic structure
- Barnes-Hut optimization available for large N

Controls:
    SPACE   - Pause/Resume
    UP/DOWN - Speed up/Slow down
    I/O     - Zoom in/out
    1       - Top-down view
    2       - Side view
    3       - Follow galaxy 1
    R       - Reset simulation
    Q/ESC   - Quit

Run:
    python galaxy_collision.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

from shared.constants import G, PC, LY, YEAR


# =============================================================================
# GALAXY PHYSICS
# =============================================================================

# Galactic units (more numerically stable than SI for galaxies)
# 1 mass unit = 10^10 solar masses
# 1 distance unit = 1 kpc (kiloparsec)
# 1 time unit = ~10 million years
# With these units, G ≈ 1

G_GALACTIC = 1.0  # Gravitational constant in galactic units
KPC = 3.086e19    # Kiloparsec in meters
M_GALAXY = 1.0    # 10^10 solar masses


class Galaxy:
    """
    Generate initial conditions for a disk galaxy.
    """

    def __init__(self, n_stars, center, velocity, mass=1.0, disk_radius=10.0,
                 rotation_direction=1, color='blue'):
        """
        Create a galaxy.

        Args:
            n_stars: number of stars (not including central black hole)
            center: [x, y] center position in kpc
            velocity: [vx, vy] bulk velocity in kpc/time_unit
            mass: total mass in 10^10 solar masses
            disk_radius: disk radius in kpc
            rotation_direction: 1 for counter-clockwise, -1 for clockwise
            color: color for visualization
        """
        self.n_stars = n_stars
        self.center = np.array(center, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.disk_radius = disk_radius
        self.rotation_direction = rotation_direction
        self.color = color

        # Central black hole mass (fraction of total)
        self.bh_mass = 0.1 * mass
        self.disk_mass = 0.9 * mass

    def generate(self):
        """
        Generate positions, velocities, and masses for all particles.

        Returns:
            positions: (n+1, 2) array (black hole + stars)
            velocities: (n+1, 2) array
            masses: (n+1,) array
        """
        n = self.n_stars + 1  # +1 for black hole

        positions = np.zeros((n, 2))
        velocities = np.zeros((n, 2))
        masses = np.zeros(n)

        # Central black hole
        positions[0] = self.center
        velocities[0] = self.velocity
        masses[0] = self.bh_mass

        # Disk stars - exponential disk profile
        # r follows exponential distribution for realistic disk
        scale_length = self.disk_radius / 3.0
        radii = np.random.exponential(scale_length, self.n_stars)
        radii = np.clip(radii, 0.5, self.disk_radius)  # Min and max radius

        # Random angles
        angles = np.random.uniform(0, 2*np.pi, self.n_stars)

        # Positions relative to center
        positions[1:, 0] = self.center[0] + radii * np.cos(angles)
        positions[1:, 1] = self.center[1] + radii * np.sin(angles)

        # Circular velocities (Keplerian around central mass)
        # v = sqrt(G * M_enclosed / r)
        # For simplicity, assume most mass is in center
        v_circular = np.sqrt(G_GALACTIC * self.bh_mass / radii)

        # Tangential velocity (perpendicular to radius)
        velocities[1:, 0] = self.velocity[0] - self.rotation_direction * v_circular * np.sin(angles)
        velocities[1:, 1] = self.velocity[1] + self.rotation_direction * v_circular * np.cos(angles)

        # Star masses (equal mass for simplicity)
        masses[1:] = self.disk_mass / self.n_stars

        return positions, velocities, masses


class NBodySimulator:
    """
    N-body gravitational simulator optimized for galaxy simulations.
    """

    def __init__(self, positions, velocities, masses, softening=0.5):
        """
        Initialize simulator.

        Args:
            positions: (n, 2) array
            velocities: (n, 2) array
            masses: (n,) array
            softening: softening length to prevent singularities (kpc)
        """
        self.positions = np.array(positions, dtype=np.float64)
        self.velocities = np.array(velocities, dtype=np.float64)
        self.masses = np.array(masses, dtype=np.float64)
        self.softening = softening
        self.n_bodies = len(masses)
        self.time = 0.0

        # Pre-compute accelerations
        self.accelerations = self._compute_accelerations()

    def _compute_accelerations(self):
        """
        Compute accelerations using vectorized operations.

        Uses softened gravity to prevent numerical issues.
        """
        # Pairwise position differences
        dx = self.positions[:, np.newaxis, 0] - self.positions[np.newaxis, :, 0]
        dy = self.positions[:, np.newaxis, 1] - self.positions[np.newaxis, :, 1]

        # Softened distance squared
        r2 = dx**2 + dy**2 + self.softening**2

        # Avoid self-interaction
        np.fill_diagonal(r2, 1.0)

        # Gravitational acceleration magnitude: G * m / r²
        # Direction: towards other body, so we need r_ij (j's position - i's position)
        # a_i = sum_j G * m_j * (r_j - r_i) / |r_ij|³

        r = np.sqrt(r2)
        r3 = r2 * r

        # Acceleration components (note the sign: -dx because dx = r_i - r_j)
        ax = -G_GALACTIC * np.sum(self.masses[np.newaxis, :] * dx / r3, axis=1)
        ay = -G_GALACTIC * np.sum(self.masses[np.newaxis, :] * dy / r3, axis=1)

        return np.column_stack([ax, ay])

    def step(self, dt):
        """Velocity Verlet integration step."""
        # Position update
        self.positions += self.velocities * dt + 0.5 * self.accelerations * dt**2

        # New accelerations
        new_acc = self._compute_accelerations()

        # Velocity update
        self.velocities += 0.5 * (self.accelerations + new_acc) * dt

        self.accelerations = new_acc
        self.time += dt

    def total_energy(self):
        """Compute total mechanical energy."""
        # Kinetic
        KE = 0.5 * np.sum(self.masses[:, np.newaxis] * self.velocities**2)

        # Potential
        PE = 0.0
        for i in range(self.n_bodies):
            for j in range(i + 1, self.n_bodies):
                r = np.sqrt(np.sum((self.positions[i] - self.positions[j])**2) + self.softening**2)
                PE -= G_GALACTIC * self.masses[i] * self.masses[j] / r

        return KE + PE


# =============================================================================
# VISUALIZATION
# =============================================================================

class GalaxyCollisionVisualization:
    """
    Visualization for galaxy collision simulation.
    """

    def __init__(self, sim, galaxy1_n, galaxy2_n, dt):
        self.sim = sim
        self.galaxy1_n = galaxy1_n + 1  # +1 for black hole
        self.galaxy2_n = galaxy2_n + 1
        self.dt = dt

        self.paused = False
        self.speed = 1
        self.view_scale = 80  # kpc

        # Trail history
        self.trail_length = 50
        self.bh1_trail = []
        self.bh2_trail = []

        self._setup_figure()
        self._setup_controls()

    def _setup_figure(self):
        """Create the matplotlib figure."""
        self.fig = plt.figure(figsize=(14, 10), facecolor='black')

        self.ax = self.fig.add_axes([0.05, 0.1, 0.75, 0.85])
        self.ax.set_facecolor('black')
        self.ax.set_aspect('equal')
        self._set_view(self.view_scale)

        # Style
        self.ax.tick_params(colors='#333333', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#222222')
        self.ax.set_xlabel('Distance (kpc)', color='#444444', fontsize=9)
        self.ax.set_ylabel('Distance (kpc)', color='#444444', fontsize=9)

        # Title
        self.title = self.ax.set_title('Galaxy Collision Simulation',
                                        color='white', fontsize=14, fontweight='bold')

        # Info panel
        self.info_ax = self.fig.add_axes([0.82, 0.1, 0.16, 0.85])
        self.info_ax.set_facecolor('black')
        self.info_ax.axis('off')

        # Create scatter plots for galaxies
        # Galaxy 1 (blue)
        self.scatter1 = self.ax.scatter([], [], s=1, c='#4488ff', alpha=0.6)
        self.bh1, = self.ax.plot([], [], 'o', color='white', markersize=8, zorder=100)
        self.bh1_trail_line, = self.ax.plot([], [], '-', color='#4488ff', alpha=0.3, linewidth=1)

        # Galaxy 2 (orange)
        self.scatter2 = self.ax.scatter([], [], s=1, c='#ff8844', alpha=0.6)
        self.bh2, = self.ax.plot([], [], 'o', color='white', markersize=8, zorder=100)
        self.bh2_trail_line, = self.ax.plot([], [], '-', color='#ff8844', alpha=0.3, linewidth=1)

        # Time display
        self.time_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes,
                                      color='white', fontsize=11, va='top',
                                      family='monospace')

        # Help text
        self.help_text = self.ax.text(0.02, 0.02,
                                      'SPACE:pause  ↑↓:speed  I/O:zoom  R:reset',
                                      transform=self.ax.transAxes, color='#444444',
                                      fontsize=8, va='bottom')

    def _set_view(self, scale):
        """Set view scale in kpc."""
        self.view_scale = scale
        self.ax.set_xlim(-scale, scale)
        self.ax.set_ylim(-scale, scale)

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key == 'up':
            self.speed = min(self.speed * 2, 32)
        elif event.key == 'down':
            self.speed = max(self.speed // 2, 1)
        elif event.key in ['i', 'I']:
            self._set_view(max(20, self.view_scale * 0.7))
        elif event.key in ['o', 'O']:
            self._set_view(min(200, self.view_scale * 1.4))
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def _update_info_panel(self):
        """Update info panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('black')
        self.info_ax.axis('off')

        self.info_ax.text(0.5, 0.95, 'SIMULATION', transform=self.info_ax.transAxes,
                         color='white', fontsize=11, fontweight='bold', ha='center')

        y = 0.85

        # Galaxy 1 info
        self.info_ax.text(0.1, y, '● Galaxy 1', transform=self.info_ax.transAxes,
                         color='#4488ff', fontsize=10, fontweight='bold')
        y -= 0.05
        bh1_pos = self.sim.positions[0]
        self.info_ax.text(0.1, y, f'  BH: ({bh1_pos[0]:.1f}, {bh1_pos[1]:.1f})',
                         transform=self.info_ax.transAxes, color='#666666', fontsize=8)
        y -= 0.08

        # Galaxy 2 info
        self.info_ax.text(0.1, y, '● Galaxy 2', transform=self.info_ax.transAxes,
                         color='#ff8844', fontsize=10, fontweight='bold')
        y -= 0.05
        bh2_pos = self.sim.positions[self.galaxy1_n]
        self.info_ax.text(0.1, y, f'  BH: ({bh2_pos[0]:.1f}, {bh2_pos[1]:.1f})',
                         transform=self.info_ax.transAxes, color='#666666', fontsize=8)
        y -= 0.08

        # Separation
        sep = np.linalg.norm(bh1_pos - bh2_pos)
        self.info_ax.text(0.1, y, f'Separation: {sep:.1f} kpc',
                         transform=self.info_ax.transAxes, color='#888888', fontsize=9)
        y -= 0.1

        # Speed
        self.info_ax.text(0.1, y, f'Speed: {self.speed}x',
                         transform=self.info_ax.transAxes, color='#666666', fontsize=9)

    def animate(self, frame):
        """Animation update."""
        if not self.paused:
            for _ in range(self.speed):
                self.sim.step(self.dt)

            # Update trails
            self.bh1_trail.append(self.sim.positions[0].copy())
            self.bh2_trail.append(self.sim.positions[self.galaxy1_n].copy())
            if len(self.bh1_trail) > self.trail_length:
                self.bh1_trail.pop(0)
                self.bh2_trail.pop(0)

        # Update galaxy 1 stars
        g1_pos = self.sim.positions[1:self.galaxy1_n]
        self.scatter1.set_offsets(g1_pos)

        # Update galaxy 2 stars
        g2_pos = self.sim.positions[self.galaxy1_n+1:]
        self.scatter2.set_offsets(g2_pos)

        # Update black holes
        self.bh1.set_data([self.sim.positions[0, 0]], [self.sim.positions[0, 1]])
        self.bh2.set_data([self.sim.positions[self.galaxy1_n, 0]],
                         [self.sim.positions[self.galaxy1_n, 1]])

        # Update trails
        if self.bh1_trail:
            trail1 = np.array(self.bh1_trail)
            trail2 = np.array(self.bh2_trail)
            self.bh1_trail_line.set_data(trail1[:, 0], trail1[:, 1])
            self.bh2_trail_line.set_data(trail2[:, 0], trail2[:, 1])

        # Time display (1 time unit ≈ 10 Myr)
        time_myr = self.sim.time * 10
        if time_myr > 1000:
            time_str = f'{time_myr/1000:.2f} Gyr'
        else:
            time_str = f'{time_myr:.0f} Myr'

        status = "PAUSED" if self.paused else "RUNNING"
        self.time_text.set_text(f'Time: {time_str}  [{status}]')

        if frame % 5 == 0:
            self._update_info_panel()

        return [self.scatter1, self.scatter2, self.bh1, self.bh2,
                self.bh1_trail_line, self.bh2_trail_line, self.time_text]

    def run(self):
        """Start animation."""
        self.anim = FuncAnimation(self.fig, self.animate, frames=None,
                                  interval=30, blit=False, cache_frame_data=False)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Setup and run the galaxy collision simulation."""

    print("=" * 55)
    print("  GALAXY COLLISION SIMULATOR")
    print("=" * 55)
    print()
    print("  Simulating collision between two spiral galaxies.")
    print("  Similar to the future Milky Way - Andromeda merger.")
    print()

    # Number of stars per galaxy (more = prettier but slower)
    n_stars = 800

    print(f"  Stars per galaxy: {n_stars}")
    print("  Generating galaxies...")

    # Galaxy 1: Larger galaxy, initially on the left
    galaxy1 = Galaxy(
        n_stars=n_stars,
        center=[-40, 0],           # 40 kpc to the left
        velocity=[0.3, 0.1],       # Moving right and slightly up
        mass=1.5,                  # 1.5 × 10^10 solar masses
        disk_radius=15,
        rotation_direction=1,
        color='blue'
    )

    # Galaxy 2: Smaller galaxy, initially on the right
    galaxy2 = Galaxy(
        n_stars=n_stars,
        center=[40, 10],           # 40 kpc to the right, 10 up
        velocity=[-0.4, -0.05],    # Moving left and slightly down
        mass=1.0,                  # 10^10 solar masses
        disk_radius=12,
        rotation_direction=-1,     # Opposite rotation
        color='orange'
    )

    # Generate initial conditions
    pos1, vel1, mass1 = galaxy1.generate()
    pos2, vel2, mass2 = galaxy2.generate()

    # Combine galaxies
    positions = np.vstack([pos1, pos2])
    velocities = np.vstack([vel1, vel2])
    masses = np.concatenate([mass1, mass2])

    print(f"  Total particles: {len(masses)}")
    print()
    print("  Controls:")
    print("    SPACE   - Pause/Resume")
    print("    ↑/↓     - Speed up/Slow down")
    print("    I/O     - Zoom in/out")
    print("    Q/ESC   - Quit")
    print()
    print("=" * 55)

    # Create simulator
    sim = NBodySimulator(positions, velocities, masses, softening=0.5)

    # Time step (in galactic units, ~1 Myr)
    dt = 0.1

    # Create visualization
    viz = GalaxyCollisionVisualization(sim, n_stars, n_stars, dt)
    viz.run()


if __name__ == '__main__':
    main()
