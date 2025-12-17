"""
Hohmann Transfer Calculator
===========================

Calculate and visualize minimum-energy transfer orbits between planets.

A Hohmann transfer is the most fuel-efficient way to travel between two
circular orbits. It uses two engine burns:
1. At departure: accelerate to enter elliptical transfer orbit
2. At arrival: accelerate again to match destination orbit

Physics:
- Vis-viva equation: v² = GM(2/r - 1/a)
- Transfer orbit is an ellipse tangent to both orbits
- Total Δv = |Δv₁| + |Δv₂|

Controls:
    LEFT/RIGHT - Select departure planet
    UP/DOWN    - Select destination planet
    ENTER      - Launch transfer animation
    R          - Reset
    Q/ESC      - Quit

Run:
    python hohmann_transfer.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.animation import FuncAnimation
import matplotlib.patheffects as pe

from shared.constants import G, AU, DAY, M_SUN, PLANETS


# =============================================================================
# ORBITAL MECHANICS
# =============================================================================

class OrbitalMechanics:
    """
    Orbital mechanics calculations for Hohmann transfers.

    All calculations assume circular, coplanar orbits (good approximation
    for inner solar system).
    """

    def __init__(self, central_mass=M_SUN):
        self.mu = G * central_mass  # Standard gravitational parameter

    def circular_velocity(self, r):
        """
        Orbital velocity for circular orbit at radius r.

        v = √(μ/r)
        """
        return np.sqrt(self.mu / r)

    def vis_viva(self, r, a):
        """
        Velocity at radius r in orbit with semi-major axis a.

        Vis-viva equation: v² = μ(2/r - 1/a)
        """
        return np.sqrt(self.mu * (2/r - 1/a))

    def orbital_period(self, a):
        """
        Orbital period for semi-major axis a.

        Kepler's 3rd law: T = 2π√(a³/μ)
        """
        return 2 * np.pi * np.sqrt(a**3 / self.mu)

    def hohmann_transfer(self, r1, r2):
        """
        Calculate Hohmann transfer parameters.

        Args:
            r1: radius of departure orbit
            r2: radius of arrival orbit

        Returns:
            dict with transfer parameters
        """
        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Velocities in circular orbits
        v1_circular = self.circular_velocity(r1)
        v2_circular = self.circular_velocity(r2)

        # Velocities in transfer orbit at departure and arrival
        v1_transfer = self.vis_viva(r1, a_transfer)
        v2_transfer = self.vis_viva(r2, a_transfer)

        # Delta-v for each burn
        dv1 = v1_transfer - v1_circular  # Departure burn
        dv2 = v2_circular - v2_transfer  # Arrival burn

        # Transfer time (half the orbital period of transfer ellipse)
        transfer_time = self.orbital_period(a_transfer) / 2

        # Phase angle: where must target planet be at launch?
        # The target needs to "meet" the spacecraft at arrival
        target_angular_velocity = self.circular_velocity(r2) / r2
        angle_traveled_by_target = target_angular_velocity * transfer_time
        phase_angle = np.pi - angle_traveled_by_target  # Radians ahead of departure

        return {
            'a_transfer': a_transfer,
            'v1_circular': v1_circular,
            'v2_circular': v2_circular,
            'v1_transfer': v1_transfer,
            'v2_transfer': v2_transfer,
            'dv1': dv1,
            'dv2': dv2,
            'total_dv': abs(dv1) + abs(dv2),
            'transfer_time': transfer_time,
            'phase_angle': phase_angle,
            'is_outward': r2 > r1
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

class HohmannVisualizer:
    """
    Interactive Hohmann transfer visualization.
    """

    PLANET_ORDER = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter']

    PLANET_COLORS = {
        'Mercury': '#888888',
        'Venus': '#e6c87a',
        'Earth': '#6b93d6',
        'Mars': '#c1440e',
        'Jupiter': '#d8ca9d'
    }

    def __init__(self):
        self.mechanics = OrbitalMechanics()
        self.departure_idx = 2  # Earth
        self.destination_idx = 3  # Mars

        self.animating = False
        self.animation = None

        self._setup_figure()
        self._setup_controls()
        self._update_display()

    def _setup_figure(self):
        """Create the matplotlib figure."""
        self.fig = plt.figure(figsize=(16, 9), facecolor='black')

        # Main orbit display
        self.ax = self.fig.add_axes([0.02, 0.05, 0.6, 0.9])
        self.ax.set_facecolor('#0a0a15')
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Info panel
        self.info_ax = self.fig.add_axes([0.65, 0.05, 0.33, 0.9])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('HOHMANN TRANSFER CALCULATOR',
                         color='white', fontsize=16, fontweight='bold', y=0.98)

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        """Handle keyboard input."""
        if self.animating:
            return

        if event.key == 'left':
            self.departure_idx = max(0, self.departure_idx - 1)
            if self.departure_idx == self.destination_idx:
                self.departure_idx = max(0, self.departure_idx - 1)
        elif event.key == 'right':
            self.departure_idx = min(len(self.PLANET_ORDER) - 1, self.departure_idx + 1)
            if self.departure_idx == self.destination_idx:
                self.departure_idx = min(len(self.PLANET_ORDER) - 1, self.departure_idx + 1)
        elif event.key == 'up':
            self.destination_idx = min(len(self.PLANET_ORDER) - 1, self.destination_idx + 1)
            if self.destination_idx == self.departure_idx:
                self.destination_idx = min(len(self.PLANET_ORDER) - 1, self.destination_idx + 1)
        elif event.key == 'down':
            self.destination_idx = max(0, self.destination_idx - 1)
            if self.destination_idx == self.departure_idx:
                self.destination_idx = max(0, self.destination_idx - 1)
        elif event.key == 'enter':
            self._start_animation()
            return
        elif event.key in ['r', 'R']:
            self.departure_idx = 2
            self.destination_idx = 3
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)
            return

        # Ensure valid selection
        self.departure_idx = max(0, min(len(self.PLANET_ORDER) - 1, self.departure_idx))
        self.destination_idx = max(0, min(len(self.PLANET_ORDER) - 1, self.destination_idx))
        if self.departure_idx == self.destination_idx:
            self.destination_idx = (self.departure_idx + 1) % len(self.PLANET_ORDER)

        self._update_display()

    def _draw_orbits(self):
        """Draw planetary orbits and the transfer ellipse."""
        self.ax.clear()
        self.ax.set_facecolor('#0a0a15')
        self.ax.axis('off')

        departure = self.PLANET_ORDER[self.departure_idx]
        destination = self.PLANET_ORDER[self.destination_idx]

        r1 = PLANETS[departure]['distance']
        r2 = PLANETS[destination]['distance']

        # Calculate transfer parameters
        transfer = self.mechanics.hohmann_transfer(r1, r2)

        # Determine view scale
        max_r = max(r1, r2) * 1.3
        self.ax.set_xlim(-max_r, max_r)
        self.ax.set_ylim(-max_r, max_r)

        # Draw Sun
        sun = Circle((0, 0), max_r * 0.03, color='yellow', zorder=10)
        self.ax.add_patch(sun)
        self.ax.text(0, -max_r * 0.08, 'Sun', color='yellow',
                    ha='center', fontsize=9)

        # Draw planetary orbits
        for planet in self.PLANET_ORDER:
            r = PLANETS[planet]['distance']
            if r <= max_r:
                orbit = Circle((0, 0), r, fill=False,
                              color=self.PLANET_COLORS[planet],
                              linestyle='--', alpha=0.4, linewidth=1)
                self.ax.add_patch(orbit)

        # Highlight departure and destination orbits
        orbit1 = Circle((0, 0), r1, fill=False,
                        color=self.PLANET_COLORS[departure],
                        linewidth=2, alpha=0.8)
        orbit2 = Circle((0, 0), r2, fill=False,
                        color=self.PLANET_COLORS[destination],
                        linewidth=2, alpha=0.8)
        self.ax.add_patch(orbit1)
        self.ax.add_patch(orbit2)

        # Draw departure planet at 0 degrees
        self.ax.plot(r1, 0, 'o', color=self.PLANET_COLORS[departure],
                    markersize=12, zorder=15)
        self.ax.text(r1 * 1.1, 0, departure, color=self.PLANET_COLORS[departure],
                    fontsize=10, fontweight='bold', va='center')

        # Draw destination planet at phase angle position
        phase = transfer['phase_angle']
        x2 = r2 * np.cos(phase)
        y2 = r2 * np.sin(phase)
        self.ax.plot(x2, y2, 'o', color=self.PLANET_COLORS[destination],
                    markersize=12, zorder=15)
        self.ax.text(x2 * 1.1, y2 * 1.1, destination,
                    color=self.PLANET_COLORS[destination],
                    fontsize=10, fontweight='bold', va='center')

        # Draw transfer orbit (ellipse)
        a = transfer['a_transfer']
        c = abs(r2 - r1) / 2  # Distance from center to focus
        b = np.sqrt(a**2 - c**2)  # Semi-minor axis

        # Ellipse center (Sun is at one focus)
        if transfer['is_outward']:
            cx = a - r1  # Center is shifted right
        else:
            cx = r1 - a  # Center is shifted left

        theta = np.linspace(0, np.pi, 100)
        if transfer['is_outward']:
            x = cx + a * np.cos(theta)
            y = b * np.sin(theta)
        else:
            x = cx - a * np.cos(theta)
            y = b * np.sin(theta)

        self.ax.plot(x, y, '-', color='#00ff88', linewidth=2.5,
                    label='Transfer orbit', zorder=5)

        # Draw Δv arrows
        arrow_scale = max_r * 0.15

        # Departure Δv
        if transfer['is_outward']:
            self.ax.annotate('', xy=(r1, arrow_scale), xytext=(r1, 0),
                           arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2))
            self.ax.text(r1 + max_r * 0.05, arrow_scale * 0.5, f'Δv₁',
                        color='#ff4444', fontsize=10, fontweight='bold')
        else:
            self.ax.annotate('', xy=(r1, -arrow_scale), xytext=(r1, 0),
                           arrowprops=dict(arrowstyle='->', color='#ff4444', lw=2))
            self.ax.text(r1 + max_r * 0.05, -arrow_scale * 0.5, f'Δv₁',
                        color='#ff4444', fontsize=10, fontweight='bold')

        # Store transfer data for info panel
        self.current_transfer = transfer
        self.departure_name = departure
        self.destination_name = destination

    def _update_info_panel(self):
        """Update the information panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        transfer = self.current_transfer

        # Title
        self.info_ax.text(0.5, 0.95, 'MISSION PARAMETERS',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=14, fontweight='bold',
                         ha='center', va='top')

        # Mission route
        direction = '→' if transfer['is_outward'] else '→'
        route = f"{self.departure_name} {direction} {self.destination_name}"
        self.info_ax.text(0.5, 0.88, route, transform=self.info_ax.transAxes,
                         color='#00ff88', fontsize=16, fontweight='bold',
                         ha='center')

        # Separator
        self.info_ax.text(0.5, 0.82, '─' * 30, transform=self.info_ax.transAxes,
                         color='#333333', ha='center', fontsize=8)

        y = 0.75
        spacing = 0.07

        # Orbital velocities
        self.info_ax.text(0.05, y, 'ORBITAL VELOCITIES',
                         transform=self.info_ax.transAxes,
                         color='#888888', fontsize=10)
        y -= spacing * 0.7

        self.info_ax.text(0.08, y, f'{self.departure_name}:',
                         transform=self.info_ax.transAxes,
                         color=self.PLANET_COLORS[self.departure_name], fontsize=11)
        self.info_ax.text(0.6, y, f'{transfer["v1_circular"]/1000:.2f} km/s',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=11, ha='right')
        y -= spacing * 0.6

        self.info_ax.text(0.08, y, f'{self.destination_name}:',
                         transform=self.info_ax.transAxes,
                         color=self.PLANET_COLORS[self.destination_name], fontsize=11)
        self.info_ax.text(0.6, y, f'{transfer["v2_circular"]/1000:.2f} km/s',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=11, ha='right')
        y -= spacing * 1.2

        # Delta-v requirements
        self.info_ax.text(0.05, y, 'DELTA-V REQUIREMENTS',
                         transform=self.info_ax.transAxes,
                         color='#888888', fontsize=10)
        y -= spacing * 0.7

        self.info_ax.text(0.08, y, 'Departure burn (Δv₁):',
                         transform=self.info_ax.transAxes,
                         color='#ff4444', fontsize=11)
        self.info_ax.text(0.6, y, f'{abs(transfer["dv1"])/1000:.2f} km/s',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=11, ha='right')
        y -= spacing * 0.6

        self.info_ax.text(0.08, y, 'Arrival burn (Δv₂):',
                         transform=self.info_ax.transAxes,
                         color='#ff4444', fontsize=11)
        self.info_ax.text(0.6, y, f'{abs(transfer["dv2"])/1000:.2f} km/s',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=11, ha='right')
        y -= spacing * 0.6

        self.info_ax.text(0.08, y, 'Total Δv:',
                         transform=self.info_ax.transAxes,
                         color='#ffff44', fontsize=12, fontweight='bold')
        self.info_ax.text(0.6, y, f'{transfer["total_dv"]/1000:.2f} km/s',
                         transform=self.info_ax.transAxes,
                         color='#ffff44', fontsize=12, fontweight='bold', ha='right')
        y -= spacing * 1.2

        # Transfer time
        self.info_ax.text(0.05, y, 'TRANSFER TIME',
                         transform=self.info_ax.transAxes,
                         color='#888888', fontsize=10)
        y -= spacing * 0.7

        days = transfer['transfer_time'] / DAY
        if days > 365:
            time_str = f'{days/365.25:.1f} years ({days:.0f} days)'
        else:
            time_str = f'{days:.0f} days'

        self.info_ax.text(0.08, y, time_str,
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=12)
        y -= spacing * 1.2

        # Phase angle
        self.info_ax.text(0.05, y, 'LAUNCH WINDOW',
                         transform=self.info_ax.transAxes,
                         color='#888888', fontsize=10)
        y -= spacing * 0.7

        phase_deg = np.degrees(transfer['phase_angle'])
        self.info_ax.text(0.08, y, f'Phase angle: {phase_deg:.1f}°',
                         transform=self.info_ax.transAxes,
                         color='white', fontsize=11)
        y -= spacing * 0.6

        self.info_ax.text(0.08, y, f'(Target leads by {abs(phase_deg):.1f}°)',
                         transform=self.info_ax.transAxes,
                         color='#666666', fontsize=9)
        y -= spacing * 1.5

        # Controls
        self.info_ax.text(0.5, 0.12, '─' * 30, transform=self.info_ax.transAxes,
                         color='#333333', ha='center', fontsize=8)

        self.info_ax.text(0.5, 0.08, 'CONTROLS', transform=self.info_ax.transAxes,
                         color='#555555', fontsize=10, ha='center')
        self.info_ax.text(0.5, 0.04, '←→: Departure  ↑↓: Destination  ENTER: Launch',
                         transform=self.info_ax.transAxes,
                         color='#444444', fontsize=8, ha='center')

    def _update_display(self):
        """Update the entire display."""
        self._draw_orbits()
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def _start_animation(self):
        """Animate the transfer."""
        if self.animating:
            return

        self.animating = True

        departure = self.PLANET_ORDER[self.departure_idx]
        destination = self.PLANET_ORDER[self.destination_idx]

        r1 = PLANETS[departure]['distance']
        r2 = PLANETS[destination]['distance']
        transfer = self.mechanics.hohmann_transfer(r1, r2)

        # Transfer ellipse parameters
        a = transfer['a_transfer']
        c = abs(r2 - r1) / 2
        b = np.sqrt(a**2 - c**2)

        if transfer['is_outward']:
            cx = a - r1
        else:
            cx = r1 - a

        # Spacecraft marker
        spacecraft, = self.ax.plot([], [], 'o', color='white', markersize=8, zorder=20)
        trail, = self.ax.plot([], [], '-', color='white', alpha=0.5, linewidth=1, zorder=18)

        trail_x = []
        trail_y = []
        n_frames = 150

        def animate(frame):
            progress = frame / n_frames

            # Parametric angle along transfer ellipse
            if transfer['is_outward']:
                theta = progress * np.pi
                x = cx + a * np.cos(theta)
                y = b * np.sin(theta)
            else:
                theta = progress * np.pi
                x = cx - a * np.cos(theta)
                y = b * np.sin(theta)

            spacecraft.set_data([x], [y])

            trail_x.append(x)
            trail_y.append(y)
            trail.set_data(trail_x, trail_y)

            if frame == n_frames - 1:
                self.animating = False

            return spacecraft, trail

        self.animation = FuncAnimation(self.fig, animate, frames=n_frames,
                                       interval=30, blit=True, repeat=False)
        self.fig.canvas.draw()

    def run(self):
        """Start the visualization."""
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 50)
    print("  HOHMANN TRANSFER CALCULATOR")
    print("=" * 50)
    print()
    print("  Calculate minimum-energy transfers between planets.")
    print()
    print("  Controls:")
    print("    ←/→     - Select departure planet")
    print("    ↑/↓     - Select destination planet")
    print("    ENTER   - Launch transfer animation")
    print("    R       - Reset to Earth-Mars")
    print("    Q/ESC   - Quit")
    print()
    print("=" * 50)

    viz = HohmannVisualizer()
    viz.run()


if __name__ == '__main__':
    main()
