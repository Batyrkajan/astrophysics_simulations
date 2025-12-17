"""
Lagrange Points Visualizer
==========================

Visualize the 5 Lagrange points in the Sun-Earth system (or any two-body
system). These are equilibrium points where a small mass can remain
stationary relative to the two larger masses.

Physics:
- Restricted 3-body problem (test particle in 2-body system)
- Co-rotating reference frame
- Effective potential (gravity + centrifugal)
- L1, L2, L3: Unstable saddle points
- L4, L5: Stable (for mass ratio < 0.0385)

Applications:
- L1: Solar observatories (SOHO)
- L2: Space telescopes (JWST, Gaia)
- L4/L5: Trojan asteroids

Controls:
    SPACE      - Pause/Resume test particle
    CLICK      - Place test particle
    1-5        - Jump to L1-L5
    P          - Toggle potential contours
    R          - Reset
    Q/ESC      - Quit

Run:
    python lagrange_points.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.optimize import brentq

from shared.constants import G, AU, M_SUN, M_EARTH, DAY, YEAR


# =============================================================================
# LAGRANGE POINT PHYSICS
# =============================================================================

class LagrangeSystem:
    """
    Calculate Lagrange points and dynamics in the rotating frame.

    Uses dimensionless units where:
    - Total mass M1 + M2 = 1
    - Distance between M1 and M2 = 1
    - Angular velocity Ω = 1
    """

    def __init__(self, mass_ratio=None, m1=M_SUN, m2=M_EARTH):
        """
        Initialize the system.

        Args:
            mass_ratio: μ = M2/(M1+M2), or computed from m1, m2
            m1, m2: masses in kg (used if mass_ratio not given)
        """
        if mass_ratio is None:
            self.mu = m2 / (m1 + m2)
        else:
            self.mu = mass_ratio

        # Position of masses in rotating frame (centered on barycenter)
        self.m1_pos = np.array([-self.mu, 0])      # Larger mass
        self.m2_pos = np.array([1 - self.mu, 0])   # Smaller mass

        # Calculate Lagrange points
        self._compute_lagrange_points()

    def _compute_lagrange_points(self):
        """Numerically find all 5 Lagrange points."""
        mu = self.mu

        # L1: Between the two masses
        # Solve: x - (1-μ)/|x+μ|² + μ/|x-1+μ|² = 0 for x in (-μ, 1-μ)
        def L1_equation(x):
            return x - (1-mu)/abs(x + mu)**2 * np.sign(x + mu) + mu/abs(x - 1 + mu)**2 * np.sign(x - 1 + mu)

        # More stable formulation
        def L1_eq(x):
            r1 = x + mu
            r2 = x - 1 + mu
            return x - (1-mu)/(r1**2) + mu/(r2**2)

        x_L1 = brentq(L1_eq, -mu + 0.01, 1 - mu - 0.01)
        self.L1 = np.array([x_L1, 0])

        # L2: Beyond the smaller mass
        def L2_eq(x):
            r1 = x + mu
            r2 = x - 1 + mu
            return x - (1-mu)/(r1**2) - mu/(r2**2)

        x_L2 = brentq(L2_eq, 1 - mu + 0.01, 2.0)
        self.L2 = np.array([x_L2, 0])

        # L3: Beyond the larger mass
        def L3_eq(x):
            r1 = abs(x + mu)
            r2 = abs(x - 1 + mu)
            return x + (1-mu)/(r1**2) + mu/(r2**2)

        x_L3 = brentq(L3_eq, -2.0, -mu - 0.01)
        self.L3 = np.array([x_L3, 0])

        # L4 and L5: Equilateral triangle points
        self.L4 = np.array([0.5 - mu, np.sqrt(3)/2])
        self.L5 = np.array([0.5 - mu, -np.sqrt(3)/2])

        self.lagrange_points = {
            'L1': self.L1,
            'L2': self.L2,
            'L3': self.L3,
            'L4': self.L4,
            'L5': self.L5
        }

    def effective_potential(self, x, y):
        """
        Effective potential in the rotating frame.

        Ω_eff = -1/2(x² + y²) - (1-μ)/r1 - μ/r2

        (Actually we use -Ω_eff so minima are equilibrium points)
        """
        r1 = np.sqrt((x + self.mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2)

        # Avoid division by zero
        r1 = np.maximum(r1, 1e-10)
        r2 = np.maximum(r2, 1e-10)

        # Effective potential (negative so equilibria are local minima for L4/L5)
        return -0.5 * (x**2 + y**2) - (1 - self.mu)/r1 - self.mu/r2

    def acceleration(self, pos, vel):
        """
        Acceleration in the rotating frame.

        Includes:
        - Gravitational attraction to both masses
        - Centrifugal force
        - Coriolis force

        In rotating frame: a = -∇Ω_eff + 2(ẏ, -ẋ)
        """
        x, y = pos
        vx, vy = vel

        r1 = np.sqrt((x + self.mu)**2 + y**2)
        r2 = np.sqrt((x - 1 + self.mu)**2 + y**2)

        # Gravitational + centrifugal
        ax = x - (1 - self.mu) * (x + self.mu) / r1**3 - self.mu * (x - 1 + self.mu) / r2**3
        ay = y - (1 - self.mu) * y / r1**3 - self.mu * y / r2**3

        # Coriolis force: 2Ω × v = 2(ẏ, -ẋ) for Ω = ẑ
        ax += 2 * vy
        ay -= 2 * vx

        return np.array([ax, ay])


class TestParticle:
    """
    Test particle to demonstrate dynamics near Lagrange points.
    """

    def __init__(self, system, position, velocity=None):
        self.system = system
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity if velocity is not None else [0, 0], dtype=float)
        self.history = [self.position.copy()]
        self.max_history = 2000

    def step(self, dt):
        """RK4 integration step."""
        def derivatives(pos, vel):
            acc = self.system.acceleration(pos, vel)
            return vel, acc

        # RK4
        k1_v, k1_a = derivatives(self.position, self.velocity)
        k2_v, k2_a = derivatives(self.position + 0.5*dt*k1_v, self.velocity + 0.5*dt*k1_a)
        k3_v, k3_a = derivatives(self.position + 0.5*dt*k2_v, self.velocity + 0.5*dt*k2_a)
        k4_v, k4_a = derivatives(self.position + dt*k3_v, self.velocity + dt*k3_a)

        self.position += dt/6 * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        self.velocity += dt/6 * (k1_a + 2*k2_a + 2*k3_a + k4_a)

        self.history.append(self.position.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)


# =============================================================================
# VISUALIZATION
# =============================================================================

class LagrangeVisualizer:
    """
    Interactive visualization of Lagrange points.
    """

    def __init__(self):
        self.system = LagrangeSystem()  # Sun-Earth by default
        self.particle = None
        self.paused = True
        self.show_potential = True

        self._setup_figure()
        self._setup_controls()
        self._draw_static_elements()

    def _setup_figure(self):
        """Create figure and axes."""
        self.fig = plt.figure(figsize=(14, 10), facecolor='#0a0a15')

        # Main axes
        self.ax = self.fig.add_axes([0.05, 0.1, 0.65, 0.85])
        self.ax.set_facecolor('#0a0a15')
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-1.8, 1.8)
        self.ax.set_ylim(-1.5, 1.5)

        # Style
        self.ax.tick_params(colors='#444444', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333333')

        # Info panel
        self.info_ax = self.fig.add_axes([0.72, 0.1, 0.26, 0.85])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('LAGRANGE POINTS - Sun-Earth System',
                         color='white', fontsize=14, fontweight='bold', y=0.97)

    def _setup_controls(self):
        """Setup keyboard and mouse controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
        elif event.key in ['1', '2', '3', '4', '5']:
            L_name = f'L{event.key}'
            pos = self.system.lagrange_points[L_name].copy()
            # Add small perturbation
            pos += np.random.randn(2) * 0.01
            self.particle = TestParticle(self.system, pos)
            self.paused = False
        elif event.key in ['p', 'P']:
            self.show_potential = not self.show_potential
            self._draw_static_elements()
        elif event.key in ['r', 'R']:
            self.particle = None
            self.paused = True
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def _on_click(self, event):
        if event.inaxes == self.ax:
            self.particle = TestParticle(self.system, [event.xdata, event.ydata])
            self.paused = False

    def _draw_static_elements(self):
        """Draw the static background elements."""
        self.ax.clear()
        self.ax.set_facecolor('#0a0a15')
        self.ax.set_xlim(-1.8, 1.8)
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_aspect('equal')
        self.ax.tick_params(colors='#444444', labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color('#333333')

        # Effective potential contours
        if self.show_potential:
            x = np.linspace(-1.8, 1.8, 300)
            y = np.linspace(-1.5, 1.5, 250)
            X, Y = np.meshgrid(x, y)
            Z = self.system.effective_potential(X, Y)

            # Clip extreme values for visualization
            Z = np.clip(Z, -5, -1)

            # Contour plot
            levels = np.linspace(-4, -1.2, 30)
            self.ax.contour(X, Y, Z, levels=levels, colors='#334455',
                           linewidths=0.5, alpha=0.5)

        # Draw orbit of smaller mass (Earth)
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1  # Normalized distance
        orbit_x = (1 - self.system.mu) * np.cos(theta) - self.system.mu
        orbit_y = (1 - self.system.mu) * np.sin(theta)
        # Actually simpler: the orbit is a unit circle in inertial frame, but we're in rotating frame
        # In rotating frame, M2 is fixed at (1-μ, 0)
        self.ax.plot([0], [0], '+', color='#444444', markersize=10)  # Barycenter

        # Draw masses
        # M1 (Sun)
        sun = Circle(self.system.m1_pos, 0.08, color='yellow', zorder=10)
        self.ax.add_patch(sun)
        self.ax.text(self.system.m1_pos[0], self.system.m1_pos[1] - 0.15,
                    'Sun', color='yellow', ha='center', fontsize=9)

        # M2 (Earth)
        earth = Circle(self.system.m2_pos, 0.04, color='#6699ff', zorder=10)
        self.ax.add_patch(earth)
        self.ax.text(self.system.m2_pos[0], self.system.m2_pos[1] - 0.12,
                    'Earth', color='#6699ff', ha='center', fontsize=9)

        # Draw Lagrange points
        L_colors = {
            'L1': '#ff6666', 'L2': '#ff6666', 'L3': '#ff6666',  # Unstable
            'L4': '#66ff66', 'L5': '#66ff66'  # Stable
        }

        for name, pos in self.system.lagrange_points.items():
            color = L_colors[name]
            self.ax.plot(pos[0], pos[1], 'o', color=color, markersize=10, zorder=15)
            self.ax.plot(pos[0], pos[1], 'o', color='white', markersize=4, zorder=16)

            # Label offset
            offset_x = 0.1 if pos[0] < 0 else -0.1
            offset_y = 0.1 if pos[1] >= 0 else -0.1
            ha = 'right' if pos[0] < 0 else 'left'

            self.ax.text(pos[0] + offset_x, pos[1] + offset_y, name,
                        color=color, fontsize=11, fontweight='bold',
                        ha=ha, va='center')

        # Help text
        self.ax.text(0.02, 0.02,
                    'CLICK: Place particle  1-5: Jump to L point  SPACE: Pause  P: Toggle potential',
                    transform=self.ax.transAxes, color='#555555',
                    fontsize=8, va='bottom')

        # Particle trail (placeholder)
        self.trail_line, = self.ax.plot([], [], '-', color='white', alpha=0.5, linewidth=1)
        self.particle_dot, = self.ax.plot([], [], 'o', color='white', markersize=6, zorder=20)

    def _update_info_panel(self):
        """Update the info panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        y = 0.95
        self.info_ax.text(0.5, y, 'LAGRANGE POINTS', transform=self.info_ax.transAxes,
                         color='white', fontsize=12, fontweight='bold', ha='center')
        y -= 0.06

        self.info_ax.text(0.5, y, '─' * 25, transform=self.info_ax.transAxes,
                         color='#333333', ha='center', fontsize=8)
        y -= 0.05

        # Explain each point
        descriptions = {
            'L1': ('Between Sun & Earth', 'SOHO solar observatory', '#ff6666', 'Unstable'),
            'L2': ('Beyond Earth from Sun', 'JWST, Gaia', '#ff6666', 'Unstable'),
            'L3': ('Opposite side of Sun', 'Counter-Earth (fiction)', '#ff6666', 'Unstable'),
            'L4': ('60° ahead of Earth', 'Trojan asteroids', '#66ff66', 'Stable'),
            'L5': ('60° behind Earth', 'Trojan asteroids', '#66ff66', 'Stable'),
        }

        for name, (desc, use, color, stability) in descriptions.items():
            self.info_ax.text(0.05, y, f'● {name}', transform=self.info_ax.transAxes,
                             color=color, fontsize=10, fontweight='bold')
            self.info_ax.text(0.25, y, f'[{stability}]', transform=self.info_ax.transAxes,
                             color='#666666', fontsize=8)
            y -= 0.035
            self.info_ax.text(0.08, y, desc, transform=self.info_ax.transAxes,
                             color='#888888', fontsize=8)
            y -= 0.03
            self.info_ax.text(0.08, y, f'Use: {use}', transform=self.info_ax.transAxes,
                             color='#666666', fontsize=7)
            y -= 0.055

        # Particle info
        if self.particle is not None:
            y -= 0.02
            self.info_ax.text(0.5, y, '─' * 25, transform=self.info_ax.transAxes,
                             color='#333333', ha='center', fontsize=8)
            y -= 0.05

            self.info_ax.text(0.05, y, 'TEST PARTICLE', transform=self.info_ax.transAxes,
                             color='white', fontsize=10, fontweight='bold')
            y -= 0.04

            pos = self.particle.position
            vel = self.particle.velocity
            self.info_ax.text(0.08, y, f'Position: ({pos[0]:.3f}, {pos[1]:.3f})',
                             transform=self.info_ax.transAxes, color='#888888', fontsize=9)
            y -= 0.035
            self.info_ax.text(0.08, y, f'Velocity: ({vel[0]:.3f}, {vel[1]:.3f})',
                             transform=self.info_ax.transAxes, color='#888888', fontsize=9)

    def animate(self, frame):
        """Animation update."""
        if self.particle is not None and not self.paused:
            # Multiple steps per frame for speed
            for _ in range(5):
                self.particle.step(0.005)

            # Update display
            history = np.array(self.particle.history)
            self.trail_line.set_data(history[:, 0], history[:, 1])
            self.particle_dot.set_data([self.particle.position[0]],
                                       [self.particle.position[1]])

        if frame % 10 == 0:
            self._update_info_panel()

        return [self.trail_line, self.particle_dot]

    def run(self):
        """Start the visualization."""
        self._update_info_panel()
        self.anim = FuncAnimation(self.fig, self.animate, frames=None,
                                  interval=30, blit=False, cache_frame_data=False)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  LAGRANGE POINTS VISUALIZER")
    print("=" * 55)
    print()
    print("  The 5 equilibrium points in the Sun-Earth system.")
    print()
    print("  Controls:")
    print("    CLICK   - Place test particle")
    print("    1-5     - Jump to Lagrange point (with perturbation)")
    print("    SPACE   - Pause/Resume")
    print("    P       - Toggle potential contours")
    print("    R       - Reset (remove particle)")
    print("    Q/ESC   - Quit")
    print()
    print("  Watch how particles behave near stable (L4, L5) vs")
    print("  unstable (L1, L2, L3) equilibrium points!")
    print()
    print("=" * 55)

    viz = LagrangeVisualizer()
    viz.run()


if __name__ == '__main__':
    main()
