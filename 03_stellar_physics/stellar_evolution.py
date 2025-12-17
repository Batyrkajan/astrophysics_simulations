"""
Stellar Evolution Simulator
===========================

Watch a star live and die! Input a star's initial mass and observe its
complete lifecycle - from main sequence through its final fate.

Physics:
- Hydrogen fusion (PP chain, CNO cycle)
- Helium flash in low-mass stars
- Shell burning and expansion to giant phase
- Final fates: white dwarf, neutron star, or black hole

Stellar Lifecycles by Mass:
- < 0.08 M☉: Brown dwarf (never ignites)
- 0.08-0.5 M☉: Red dwarf → White dwarf
- 0.5-8 M☉: Main sequence → Red giant → Planetary nebula → White dwarf
- 8-25 M☉: Main sequence → Supergiant → Supernova → Neutron star
- > 25 M☉: Main sequence → Supergiant → Supernova → Black hole

Controls:
    UP/DOWN   - Adjust initial mass
    SPACE     - Start/Pause evolution
    R         - Reset
    1-5       - Jump to preset masses
    Q/ESC     - Quit

Run:
    python stellar_evolution.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

from shared.constants import L_SUN, T_SUN, R_SUN, M_SUN, YEAR


# =============================================================================
# STELLAR EVOLUTION PHYSICS
# =============================================================================

def main_sequence_lifetime(mass):
    """
    Main sequence lifetime in years.

    τ ≈ 10^10 * (M/M☉)^(-2.5) years

    Lower mass = longer life (less luminous, burns slower)
    """
    return 1e10 * mass**(-2.5)


def main_sequence_luminosity(mass):
    """Luminosity on main sequence."""
    if mass < 0.43:
        return 0.23 * mass**2.3
    elif mass < 2:
        return mass**4
    elif mass < 20:
        return 1.5 * mass**3.5
    else:
        return 3200 * mass


def main_sequence_temperature(mass):
    """Temperature on main sequence."""
    return T_SUN * mass**0.5


def main_sequence_radius(mass):
    """Radius on main sequence (approximate)."""
    return mass**0.8


class StellarEvolution:
    """
    Simulate the evolution of a star given its initial mass.
    """

    # Evolution phases
    PHASES = [
        'Protostar',
        'Main Sequence',
        'Subgiant',
        'Red Giant',
        'Helium Flash',
        'Horizontal Branch',
        'Asymptotic Giant',
        'Planetary Nebula',
        'White Dwarf',
        'Supergiant',
        'Supernova',
        'Neutron Star',
        'Black Hole',
        'Brown Dwarf'
    ]

    def __init__(self, initial_mass):
        """
        Initialize stellar evolution.

        Args:
            initial_mass: Initial mass in solar masses
        """
        self.initial_mass = initial_mass
        self.mass = initial_mass
        self.reset()

    def reset(self):
        """Reset to initial state."""
        self.mass = self.initial_mass
        self.age = 0
        self.phase_idx = 0
        self.phase = 'Protostar'
        self.finished = False

        # Current stellar properties
        self._update_properties()

        # Evolution track for HR diagram
        self.track_T = [self.temperature]
        self.track_L = [self.luminosity]
        self.track_phase = [self.phase]

        # Determine evolution path based on mass
        self._determine_fate()

    def _determine_fate(self):
        """Determine the star's evolutionary path based on initial mass."""
        M = self.initial_mass

        if M < 0.08:
            # Brown dwarf
            self.path = ['Protostar', 'Brown Dwarf']
            self.fate = 'Brown Dwarf'
        elif M < 0.5:
            # Low mass: very long main sequence, then white dwarf
            self.path = ['Protostar', 'Main Sequence', 'Red Giant', 'White Dwarf']
            self.fate = 'White Dwarf'
        elif M < 8:
            # Sun-like: full evolution to white dwarf
            self.path = ['Protostar', 'Main Sequence', 'Subgiant', 'Red Giant',
                        'Horizontal Branch', 'Asymptotic Giant',
                        'Planetary Nebula', 'White Dwarf']
            self.fate = 'White Dwarf'
        elif M < 25:
            # Massive: supernova to neutron star
            self.path = ['Protostar', 'Main Sequence', 'Supergiant',
                        'Supernova', 'Neutron Star']
            self.fate = 'Neutron Star'
        else:
            # Very massive: supernova to black hole
            self.path = ['Protostar', 'Main Sequence', 'Supergiant',
                        'Supernova', 'Black Hole']
            self.fate = 'Black Hole'

    def _update_properties(self):
        """Update stellar properties based on current phase."""
        M = self.mass
        M0 = self.initial_mass

        if self.phase == 'Protostar':
            self.luminosity = 10 * main_sequence_luminosity(M0)
            self.temperature = 3000
            self.radius = 10 * main_sequence_radius(M0)

        elif self.phase == 'Brown Dwarf':
            self.luminosity = 1e-4
            self.temperature = 2000
            self.radius = 0.1

        elif self.phase == 'Main Sequence':
            self.luminosity = main_sequence_luminosity(M0)
            self.temperature = main_sequence_temperature(M0)
            self.radius = main_sequence_radius(M0)

        elif self.phase == 'Subgiant':
            self.luminosity = main_sequence_luminosity(M0) * 3
            self.temperature = main_sequence_temperature(M0) * 0.9
            self.radius = main_sequence_radius(M0) * 2

        elif self.phase == 'Red Giant':
            self.luminosity = main_sequence_luminosity(M0) * 100
            self.temperature = 4000
            self.radius = main_sequence_radius(M0) * 50

        elif self.phase == 'Horizontal Branch':
            self.luminosity = 50
            self.temperature = 5000
            self.radius = 10

        elif self.phase == 'Asymptotic Giant':
            self.luminosity = main_sequence_luminosity(M0) * 1000
            self.temperature = 3500
            self.radius = main_sequence_radius(M0) * 200

        elif self.phase == 'Planetary Nebula':
            self.luminosity = 1000
            self.temperature = 100000
            self.radius = 0.1

        elif self.phase == 'White Dwarf':
            self.luminosity = 0.01
            self.temperature = 20000
            self.radius = 0.01
            self.mass = min(1.4, M0 * 0.6)  # Chandrasekhar limit

        elif self.phase == 'Supergiant':
            if M0 > 25:
                self.luminosity = main_sequence_luminosity(M0) * 10
                self.temperature = 3500 if np.random.random() > 0.5 else 25000
            else:
                self.luminosity = main_sequence_luminosity(M0) * 5
                self.temperature = 3800
            self.radius = 500

        elif self.phase == 'Supernova':
            self.luminosity = 1e10  # Briefly as bright as a galaxy!
            self.temperature = 100000
            self.radius = 1000

        elif self.phase == 'Neutron Star':
            self.luminosity = 0.001
            self.temperature = 1e6
            self.radius = 1e-5  # ~10 km
            self.mass = 1.4

        elif self.phase == 'Black Hole':
            self.luminosity = 0
            self.temperature = 0
            self.radius = 3e-6 * M0  # Schwarzschild radius
            self.mass = M0 * 0.5

    def step(self, dt_years=1e6):
        """
        Advance evolution by dt years.

        Returns True if phase changed.
        """
        if self.finished:
            return False

        self.age += dt_years
        phase_changed = False

        # Check for phase transitions
        ms_lifetime = main_sequence_lifetime(self.initial_mass)

        # Determine current phase based on age
        current_path_idx = self.path.index(self.phase) if self.phase in self.path else 0

        # Phase transition logic
        if self.phase == 'Protostar' and self.age > 1e6:
            if self.initial_mass < 0.08:
                self.phase = 'Brown Dwarf'
            else:
                self.phase = 'Main Sequence'
            phase_changed = True

        elif self.phase == 'Brown Dwarf':
            self.finished = True

        elif self.phase == 'Main Sequence' and self.age > ms_lifetime:
            if current_path_idx + 1 < len(self.path):
                self.phase = self.path[current_path_idx + 1]
                phase_changed = True

        elif self.phase in ['Subgiant', 'Red Giant', 'Horizontal Branch', 'Asymptotic Giant']:
            # These phases are relatively quick
            phase_duration = ms_lifetime * 0.1
            if self.age > ms_lifetime + phase_duration * (current_path_idx - 1):
                if current_path_idx + 1 < len(self.path):
                    self.phase = self.path[current_path_idx + 1]
                    phase_changed = True

        elif self.phase == 'Planetary Nebula':
            if self.age > ms_lifetime * 1.5:
                self.phase = 'White Dwarf'
                phase_changed = True

        elif self.phase == 'Supergiant':
            if self.age > ms_lifetime * 1.1:
                self.phase = 'Supernova'
                phase_changed = True

        elif self.phase == 'Supernova':
            # Supernova is very brief
            self.phase = 'Neutron Star' if self.initial_mass < 25 else 'Black Hole'
            phase_changed = True

        elif self.phase in ['White Dwarf', 'Neutron Star', 'Black Hole']:
            self.finished = True

        if phase_changed:
            self._update_properties()

        # Record track
        self.track_T.append(self.temperature)
        self.track_L.append(self.luminosity)
        self.track_phase.append(self.phase)

        return phase_changed


def temperature_to_rgb(T):
    """Convert temperature to approximate color."""
    T = np.clip(T, 1000, 50000)

    if T < 6600:
        r = 1.0
        g = np.clip(0.39 * np.log(T/100) - 0.634, 0, 1)
        b = np.clip(0.543 * np.log(max(T/100 - 10, 1)) - 1.186, 0, 1) if T > 2000 else 0
    else:
        r = np.clip(1.29 * (T/100 - 60)**(-0.133), 0, 1)
        g = np.clip(1.13 * (T/100 - 60)**(-0.076), 0, 1)
        b = 1.0

    return (r, g, b)


# =============================================================================
# VISUALIZATION
# =============================================================================

class StellarEvolutionVisualizer:
    """
    Animated visualization of stellar evolution.
    """

    MASS_PRESETS = [0.5, 1.0, 2.0, 10.0, 40.0]

    def __init__(self, initial_mass=1.0):
        self.star = StellarEvolution(initial_mass)
        self.running = False
        self.speed = 1

        self._setup_figure()
        self._setup_controls()
        self._update_display()

    def _setup_figure(self):
        """Create figure."""
        self.fig = plt.figure(figsize=(16, 9), facecolor='black')

        # Star visualization (left)
        self.star_ax = self.fig.add_axes([0.02, 0.15, 0.35, 0.75])
        self.star_ax.set_facecolor('black')
        self.star_ax.set_aspect('equal')
        self.star_ax.axis('off')

        # HR diagram (center)
        self.hr_ax = self.fig.add_axes([0.4, 0.15, 0.35, 0.75])
        self.hr_ax.set_facecolor('#0a0a15')

        # Info panel (right)
        self.info_ax = self.fig.add_axes([0.78, 0.15, 0.2, 0.75])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('STELLAR EVOLUTION SIMULATOR',
                         color='white', fontsize=16, fontweight='bold', y=0.96)

        # Mass slider text
        self.mass_text = self.fig.text(0.5, 0.05,
                                       f'Initial Mass: {self.star.initial_mass:.1f} M☉',
                                       color='white', fontsize=14, ha='center')
        self.fig.text(0.5, 0.02, '↑/↓: Adjust Mass   SPACE: Start/Pause   1-5: Presets   R: Reset',
                     color='#666666', fontsize=9, ha='center')

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.running = not self.running
        elif event.key == 'up':
            new_mass = min(100, self.star.initial_mass * 1.2)
            self.star = StellarEvolution(new_mass)
            self._update_display()
        elif event.key == 'down':
            new_mass = max(0.05, self.star.initial_mass / 1.2)
            self.star = StellarEvolution(new_mass)
            self._update_display()
        elif event.key in ['1', '2', '3', '4', '5']:
            idx = int(event.key) - 1
            self.star = StellarEvolution(self.MASS_PRESETS[idx])
            self._update_display()
        elif event.key in ['r', 'R']:
            self.star.reset()
            self.running = False
            self._update_display()
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

        self.mass_text.set_text(f'Initial Mass: {self.star.initial_mass:.1f} M☉')

    def _draw_star(self):
        """Draw the star visualization."""
        self.star_ax.clear()
        self.star_ax.set_facecolor('black')
        self.star_ax.axis('off')
        self.star_ax.set_xlim(-1.5, 1.5)
        self.star_ax.set_ylim(-1.5, 1.5)

        # Get star properties
        phase = self.star.phase
        T = self.star.temperature
        R = self.star.radius
        L = self.star.luminosity

        # Normalize radius for display
        display_radius = min(1.2, 0.1 + 0.3 * np.log10(max(R, 0.001) + 1))

        color = temperature_to_rgb(T)

        # Special visualizations for certain phases
        if phase == 'Black Hole':
            # Black circle with accretion disk hint
            circle = Circle((0, 0), display_radius, color='black',
                           ec='purple', linewidth=3, zorder=10)
            self.star_ax.add_patch(circle)
            self.star_ax.text(0, -1.3, 'EVENT HORIZON', color='purple',
                             ha='center', fontsize=10, fontweight='bold')

        elif phase == 'Neutron Star':
            # Small, bright blue
            circle = Circle((0, 0), 0.15, color='#aaddff', zorder=10)
            self.star_ax.add_patch(circle)
            # Magnetic field lines
            for angle in [45, 135, 225, 315]:
                rad = np.radians(angle)
                self.star_ax.plot([0.15*np.cos(rad), 0.8*np.cos(rad)],
                                 [0.15*np.sin(rad), 0.8*np.sin(rad)],
                                 color='cyan', linewidth=2, alpha=0.5)

        elif phase == 'Supernova':
            # Explosion!
            for i in range(20):
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0.3, 1.2)
                size = np.random.uniform(0.05, 0.2)
                c = Circle((r*np.cos(angle), r*np.sin(angle)), size,
                          color=np.random.choice(['yellow', 'orange', 'red', 'white']),
                          alpha=0.8, zorder=5)
                self.star_ax.add_patch(c)
            self.star_ax.text(0, -1.3, 'SUPERNOVA!', color='yellow',
                             ha='center', fontsize=14, fontweight='bold')

        elif phase == 'Planetary Nebula':
            # Expanding shell
            for r in np.linspace(0.3, 1.0, 5):
                circle = Circle((0, 0), r, fill=False,
                               color='cyan', alpha=0.3, linewidth=2)
                self.star_ax.add_patch(circle)
            # Central white dwarf
            circle = Circle((0, 0), 0.1, color='white', zorder=10)
            self.star_ax.add_patch(circle)

        else:
            # Normal star
            # Glow effect
            for r_mult in [1.3, 1.2, 1.1]:
                glow = Circle((0, 0), display_radius * r_mult,
                             color=color, alpha=0.2, zorder=5)
                self.star_ax.add_patch(glow)

            # Main star
            circle = Circle((0, 0), display_radius, color=color, zorder=10)
            self.star_ax.add_patch(circle)

        # Phase label
        self.star_ax.text(0, 1.35, phase.upper(), color='white',
                         ha='center', fontsize=12, fontweight='bold')

    def _draw_hr_diagram(self):
        """Draw the HR diagram with evolution track."""
        self.hr_ax.clear()
        self.hr_ax.set_facecolor('#0a0a15')

        # Setup
        self.hr_ax.set_xscale('log')
        self.hr_ax.set_yscale('log')
        self.hr_ax.set_xlim(50000, 2000)
        self.hr_ax.set_ylim(1e-5, 1e7)

        # Background regions
        self.hr_ax.fill_between([50000, 2000], [1e-5, 1e-5], [0.1, 0.1],
                               color='#1a1a2e', alpha=0.5)
        self.hr_ax.text(15000, 0.003, 'White Dwarfs', color='#444466', fontsize=8)

        self.hr_ax.text(6000, 0.5, 'Main Sequence', color='#334455',
                       fontsize=8, rotation=45)
        self.hr_ax.text(4000, 500, 'Giants', color='#443333', fontsize=8)
        self.hr_ax.text(3500, 100000, 'Supergiants', color='#442222', fontsize=8)

        # Evolution track
        if len(self.star.track_T) > 1:
            T_arr = np.array(self.star.track_T)
            L_arr = np.array(self.star.track_L)

            # Color by phase
            colors = [temperature_to_rgb(T) for T in T_arr]

            # Plot track
            self.hr_ax.plot(T_arr, L_arr, '-', color='white', alpha=0.3, linewidth=1)

            # Plot points
            self.hr_ax.scatter(T_arr[:-1], L_arr[:-1], c=colors[:-1], s=5, alpha=0.5)

        # Current position
        self.hr_ax.scatter(self.star.temperature, self.star.luminosity,
                          c=[temperature_to_rgb(self.star.temperature)],
                          s=200, edgecolors='white', linewidths=2, zorder=20)

        # Labels
        self.hr_ax.set_xlabel('Temperature (K)', color='white', fontsize=10)
        self.hr_ax.set_ylabel('Luminosity (L☉)', color='white', fontsize=10)
        self.hr_ax.set_title('HR DIAGRAM', color='white', fontsize=11)
        self.hr_ax.tick_params(colors='#888888', labelsize=8)

    def _update_info_panel(self):
        """Update the info panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        y = 0.95

        self.info_ax.text(0.5, y, 'STELLAR DATA', transform=self.info_ax.transAxes,
                         color='white', fontsize=11, fontweight='bold', ha='center')
        y -= 0.08

        # Age
        age = self.star.age
        if age < 1e6:
            age_str = f'{age:.0f} years'
        elif age < 1e9:
            age_str = f'{age/1e6:.1f} Myr'
        else:
            age_str = f'{age/1e9:.2f} Gyr'

        props = [
            ('Phase', self.star.phase),
            ('Age', age_str),
            ('', ''),
            ('Mass', f'{self.star.mass:.2f} M☉'),
            ('Luminosity', f'{self.star.luminosity:.2g} L☉'),
            ('Temperature', f'{self.star.temperature:.0f} K'),
            ('Radius', f'{self.star.radius:.2g} R☉'),
        ]

        for name, val in props:
            if name:
                self.info_ax.text(0.05, y, name + ':', transform=self.info_ax.transAxes,
                                 color='#888888', fontsize=9)
                self.info_ax.text(0.95, y, str(val), transform=self.info_ax.transAxes,
                                 color='white', fontsize=9, ha='right')
            y -= 0.05

        # Fate
        y -= 0.05
        self.info_ax.text(0.5, y, '─' * 15, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.05

        fate_colors = {
            'White Dwarf': '#aaaaff',
            'Neutron Star': '#aaddff',
            'Black Hole': '#aa66ff',
            'Brown Dwarf': '#aa6633'
        }

        self.info_ax.text(0.5, y, 'FINAL FATE', transform=self.info_ax.transAxes,
                         color='#666666', fontsize=9, ha='center')
        y -= 0.05
        self.info_ax.text(0.5, y, self.star.fate, transform=self.info_ax.transAxes,
                         color=fate_colors.get(self.star.fate, 'white'),
                         fontsize=12, fontweight='bold', ha='center')

        # MS lifetime
        y -= 0.08
        ms_life = main_sequence_lifetime(self.star.initial_mass)
        if ms_life < 1e9:
            life_str = f'{ms_life/1e6:.0f} Myr'
        else:
            life_str = f'{ms_life/1e9:.1f} Gyr'
        self.info_ax.text(0.5, y, f'MS Lifetime: {life_str}', transform=self.info_ax.transAxes,
                         color='#666666', fontsize=8, ha='center')

    def _update_display(self):
        """Update all display elements."""
        self._draw_star()
        self._draw_hr_diagram()
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def animate(self, frame):
        """Animation step."""
        if self.running and not self.star.finished:
            # Time step based on stellar lifetime
            ms_life = main_sequence_lifetime(self.star.initial_mass)
            dt = ms_life / 500  # Complete evolution in ~500 frames

            self.star.step(dt)
            self._update_display()

        return []

    def run(self):
        """Start the visualization."""
        self.anim = FuncAnimation(self.fig, self.animate, frames=None,
                                  interval=50, blit=False, cache_frame_data=False)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  STELLAR EVOLUTION SIMULATOR")
    print("=" * 55)
    print()
    print("  Watch a star live and die!")
    print()
    print("  Controls:")
    print("    SPACE   - Start/Pause evolution")
    print("    ↑/↓     - Increase/Decrease mass")
    print("    1-5     - Preset masses (0.5, 1, 2, 10, 40 M☉)")
    print("    R       - Reset")
    print("    Q/ESC   - Quit")
    print()
    print("  Fates by mass:")
    print("    < 8 M☉  → White Dwarf")
    print("    8-25 M☉ → Neutron Star")
    print("    > 25 M☉ → Black Hole")
    print()
    print("=" * 55)

    viz = StellarEvolutionVisualizer(initial_mass=1.0)
    viz.run()


if __name__ == '__main__':
    main()
