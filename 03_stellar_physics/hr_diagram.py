"""
Hertzsprung-Russell Diagram
===========================

The HR diagram is the astronomer's most important tool for understanding
stars. By plotting luminosity vs temperature, patterns emerge that reveal
stellar structure and evolution.

Features:
- Main sequence, giants, supergiants, white dwarfs
- Spectral classification (O, B, A, F, G, K, M)
- Color coding by true stellar color
- Interactive exploration
- Stellar properties calculator

Physics:
- Stefan-Boltzmann law: L = 4πR²σT⁴
- Mass-luminosity relation: L ∝ M^3.5 (main sequence)
- Blackbody radiation for stellar colors

Controls:
    CLICK    - Select star, show properties
    1-5      - Highlight stellar types
    C        - Toggle constellation view
    R        - Reset view
    Q/ESC    - Quit

Run:
    python hr_diagram.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

from shared.constants import L_SUN, T_SUN, R_SUN, M_SUN, sigma


# =============================================================================
# STELLAR PHYSICS
# =============================================================================

def temperature_to_rgb(T):
    """
    Convert stellar temperature to RGB color.

    Based on blackbody radiation - approximates what color a star
    actually appears to human eyes.

    Args:
        T: temperature in Kelvin

    Returns:
        (r, g, b) tuple with values 0-1
    """
    # Normalize temperature to 1000-40000K range
    T = np.clip(T, 1000, 40000)

    # Approximate blackbody color
    # Based on CIE color matching and Planck's law
    if T < 6600:
        r = 1.0
        g = np.clip(0.39 * np.log(T/100) - 0.634, 0, 1)
        if T < 2000:
            b = 0.0
        else:
            b = np.clip(0.543 * np.log(T/100 - 10) - 1.186, 0, 1)
    else:
        r = np.clip(1.29 * (T/100 - 60)**(-0.133), 0, 1)
        g = np.clip(1.13 * (T/100 - 60)**(-0.076), 0, 1)
        b = 1.0

    return (r, g, b)


def spectral_class_to_temp(spectral_class):
    """Convert spectral class to approximate temperature."""
    classes = {
        'O': 35000, 'B': 20000, 'A': 9000, 'F': 7000,
        'G': 5500, 'K': 4500, 'M': 3200
    }
    return classes.get(spectral_class[0], 5500)


def temp_to_spectral_class(T):
    """Convert temperature to spectral class."""
    if T > 30000: return 'O'
    elif T > 10000: return 'B'
    elif T > 7500: return 'A'
    elif T > 6000: return 'F'
    elif T > 5000: return 'G'
    elif T > 3500: return 'K'
    else: return 'M'


def luminosity_to_radius(L, T):
    """
    Calculate stellar radius from luminosity and temperature.

    Stefan-Boltzmann law: L = 4πR²σT⁴
    Therefore: R = √(L / (4πσT⁴))

    Returns radius in solar radii.
    """
    L_si = L * L_SUN
    R_si = np.sqrt(L_si / (4 * np.pi * sigma * T**4))
    return R_si / R_SUN


def main_sequence_luminosity(mass):
    """
    Mass-luminosity relation for main sequence stars.

    L/L_sun ≈ (M/M_sun)^α where:
    - α ≈ 4 for M < 0.43 M_sun
    - α ≈ 3.5 for 0.43 < M < 2 M_sun
    - α ≈ 3 for M > 2 M_sun
    """
    if mass < 0.43:
        return 0.23 * mass**2.3
    elif mass < 2:
        return mass**4
    elif mass < 20:
        return 1.5 * mass**3.5
    else:
        return 3200 * mass


def main_sequence_temperature(mass):
    """Approximate temperature for main sequence stars."""
    return T_SUN * mass**0.5


# =============================================================================
# STAR DATABASE
# =============================================================================

# Famous stars with their properties
# (name, temperature K, luminosity L_sun, spectral type, notes)
FAMOUS_STARS = [
    # Main Sequence
    ('Sun', 5778, 1.0, 'G2V', 'Our star'),
    ('Sirius A', 9940, 25.4, 'A1V', 'Brightest in night sky'),
    ('Alpha Centauri A', 5790, 1.52, 'G2V', 'Nearest Sun-like star'),
    ('Vega', 9602, 40.1, 'A0V', 'Summer Triangle'),
    ('Fomalhaut', 8590, 16.6, 'A3V', 'Has exoplanet'),
    ('Altair', 7670, 10.6, 'A7V', 'Summer Triangle'),
    ('Procyon A', 6530, 6.93, 'F5IV', 'Binary system'),
    ('Tau Ceti', 5344, 0.52, 'G8V', 'Nearby, exoplanets'),

    # Blue Giants/Supergiants (O, B types)
    ('Rigel', 12100, 120000, 'B8Ia', 'Blue supergiant'),
    ('Spica', 25300, 20500, 'B1V', 'Binary system'),
    ('Regulus', 12460, 316, 'B8IV', 'Heart of the Lion'),
    ('Bellatrix', 22000, 9211, 'B2III', 'Amazonian star'),
    ('Alnilam', 27000, 537000, 'B0Ia', 'Orion\'s Belt'),

    # Red Giants
    ('Arcturus', 4286, 170, 'K1.5III', 'Brightest in north'),
    ('Aldebaran', 3900, 439, 'K5III', 'Eye of the Bull'),
    ('Pollux', 4666, 43, 'K0III', 'Gemini twin'),
    ('Capella A', 4970, 78.7, 'G8III', 'Binary giant'),

    # Red Supergiants
    ('Betelgeuse', 3500, 126000, 'M1Ia', 'Might go supernova'),
    ('Antares', 3660, 75900, 'M1Ib', 'Heart of Scorpius'),
    ('Mu Cephei', 3750, 350000, 'M2Ia', 'Garnet Star'),

    # White Dwarfs
    ('Sirius B', 25200, 0.056, 'DA2', 'Famous white dwarf'),
    ('Procyon B', 7740, 0.00049, 'DQZ', 'Faint companion'),
    ('40 Eridani B', 16500, 0.013, 'DA4', 'First known WD'),

    # Other interesting stars
    ('Canopus', 7350, 10700, 'A9II', '2nd brightest'),
    ('Polaris', 6015, 1260, 'F7Ib', 'North Star'),
    ('Deneb', 8525, 196000, 'A2Ia', 'Very luminous'),
]


def generate_main_sequence(n=200):
    """Generate random main sequence stars."""
    # Mass distribution (IMF-like)
    masses = 10**np.random.uniform(-0.5, 1.5, n)
    temperatures = main_sequence_temperature(masses)
    luminosities = np.array([main_sequence_luminosity(m) for m in masses])

    # Add some scatter
    luminosities *= 10**np.random.normal(0, 0.1, n)
    temperatures *= 10**np.random.normal(0, 0.02, n)

    return temperatures, luminosities


def generate_giants(n=50):
    """Generate random giant stars."""
    temperatures = np.random.uniform(3500, 5500, n)
    luminosities = 10**np.random.uniform(1.5, 3, n)
    return temperatures, luminosities


def generate_supergiants(n=20):
    """Generate random supergiant stars."""
    # Red supergiants
    n_red = n // 2
    temp_red = np.random.uniform(3000, 4500, n_red)
    lum_red = 10**np.random.uniform(4, 5.5, n_red)

    # Blue supergiants
    n_blue = n - n_red
    temp_blue = np.random.uniform(10000, 30000, n_blue)
    lum_blue = 10**np.random.uniform(4.5, 6, n_blue)

    return (np.concatenate([temp_red, temp_blue]),
            np.concatenate([lum_red, lum_blue]))


def generate_white_dwarfs(n=30):
    """Generate random white dwarf stars."""
    temperatures = np.random.uniform(5000, 40000, n)
    luminosities = 10**np.random.uniform(-4, -1.5, n)
    return temperatures, luminosities


# =============================================================================
# VISUALIZATION
# =============================================================================

class HRDiagram:
    """
    Interactive Hertzsprung-Russell diagram visualization.
    """

    def __init__(self):
        self.selected_star = None
        self.highlight_type = None

        self._generate_stars()
        self._setup_figure()
        self._setup_controls()
        self._draw_diagram()

    def _generate_stars(self):
        """Generate all stars for the diagram."""
        # Main sequence
        ms_temp, ms_lum = generate_main_sequence(300)

        # Giants
        g_temp, g_lum = generate_giants(60)

        # Supergiants
        sg_temp, sg_lum = generate_supergiants(25)

        # White dwarfs
        wd_temp, wd_lum = generate_white_dwarfs(40)

        # Combine
        self.bg_temps = np.concatenate([ms_temp, g_temp, sg_temp, wd_temp])
        self.bg_lums = np.concatenate([ms_lum, g_lum, sg_lum, wd_lum])
        self.bg_colors = [temperature_to_rgb(T) for T in self.bg_temps]

        # Famous stars
        self.star_names = [s[0] for s in FAMOUS_STARS]
        self.star_temps = np.array([s[1] for s in FAMOUS_STARS])
        self.star_lums = np.array([s[2] for s in FAMOUS_STARS])
        self.star_types = [s[3] for s in FAMOUS_STARS]
        self.star_notes = [s[4] for s in FAMOUS_STARS]
        self.star_colors = [temperature_to_rgb(T) for T in self.star_temps]

    def _setup_figure(self):
        """Create figure and axes."""
        self.fig = plt.figure(figsize=(14, 10), facecolor='#0a0a15')

        # Main HR diagram
        self.ax = self.fig.add_axes([0.08, 0.1, 0.6, 0.8])
        self.ax.set_facecolor('#0a0a15')

        # Info panel
        self.info_ax = self.fig.add_axes([0.72, 0.1, 0.26, 0.8])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('HERTZSPRUNG-RUSSELL DIAGRAM',
                         color='white', fontsize=16, fontweight='bold', y=0.96)

    def _setup_controls(self):
        """Setup keyboard and mouse controls."""
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_click(self, event):
        """Handle mouse clicks to select stars."""
        if event.inaxes != self.ax:
            return

        # Find nearest famous star
        x_click = np.log10(event.xdata) if event.xdata > 0 else 0
        y_click = np.log10(event.ydata) if event.ydata > 0 else 0

        x_stars = np.log10(self.star_temps)
        y_stars = np.log10(self.star_lums)

        # Normalize for distance calculation
        x_range = x_stars.max() - x_stars.min()
        y_range = y_stars.max() - y_stars.min()

        distances = np.sqrt(((x_stars - x_click)/x_range)**2 +
                           ((y_stars - y_click)/y_range)**2)

        nearest_idx = np.argmin(distances)
        if distances[nearest_idx] < 0.15:
            self.selected_star = nearest_idx
            self._update_info_panel()
            self._draw_diagram()

    def _on_key(self, event):
        """Handle keyboard input."""
        if event.key == '1':
            self.highlight_type = 'main_sequence'
        elif event.key == '2':
            self.highlight_type = 'giants'
        elif event.key == '3':
            self.highlight_type = 'supergiants'
        elif event.key == '4':
            self.highlight_type = 'white_dwarfs'
        elif event.key == '5':
            self.highlight_type = None
        elif event.key in ['r', 'R']:
            self.selected_star = None
            self.highlight_type = None
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)
            return

        self._draw_diagram()
        self._update_info_panel()

    def _draw_diagram(self):
        """Draw the HR diagram."""
        self.ax.clear()
        self.ax.set_facecolor('#0a0a15')

        # Log scales
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')

        # Invert x-axis (hot stars on left, cool on right)
        self.ax.set_xlim(50000, 2000)
        self.ax.set_ylim(1e-5, 1e7)

        # Background stars
        alphas = np.ones(len(self.bg_temps)) * 0.4
        sizes = np.ones(len(self.bg_temps)) * 3

        # Highlight specific types
        if self.highlight_type == 'main_sequence':
            for i, (T, L) in enumerate(zip(self.bg_temps, self.bg_lums)):
                # Main sequence: L ~ T^4 relationship
                expected_L = (T/T_SUN)**4
                if 0.1 < L/expected_L < 10:
                    alphas[i] = 0.8
                    sizes[i] = 6
                else:
                    alphas[i] = 0.1

        self.ax.scatter(self.bg_temps, self.bg_lums, c=self.bg_colors,
                       s=sizes, alpha=alphas, zorder=1)

        # Famous stars
        for i, (name, T, L, stype, _) in enumerate(FAMOUS_STARS):
            color = temperature_to_rgb(T)
            size = 80 + 20 * np.log10(max(L, 0.0001))
            size = np.clip(size, 40, 300)

            alpha = 0.9 if self.selected_star == i else 0.7
            edge = 'white' if self.selected_star == i else None
            lw = 2 if self.selected_star == i else 0

            self.ax.scatter(T, L, c=[color], s=size, alpha=alpha,
                           edgecolors=edge, linewidths=lw, zorder=10)

            # Label bright or selected stars
            if L > 100 or self.selected_star == i:
                self.ax.annotate(name, (T, L), textcoords='offset points',
                                xytext=(5, 5), fontsize=8, color='white',
                                alpha=0.8)

        # Mark the Sun
        sun_idx = self.star_names.index('Sun')
        self.ax.scatter(self.star_temps[sun_idx], self.star_lums[sun_idx],
                       marker='*', s=200, c='yellow', edgecolors='orange',
                       linewidths=1, zorder=15)

        # Region labels
        self.ax.text(4500, 3e5, 'SUPERGIANTS', color='#ff6666',
                    fontsize=10, fontweight='bold', alpha=0.7)
        self.ax.text(4000, 100, 'GIANTS', color='#ffaa66',
                    fontsize=10, fontweight='bold', alpha=0.7)
        self.ax.text(8000, 1, 'MAIN\nSEQUENCE', color='#66aaff',
                    fontsize=9, fontweight='bold', alpha=0.7, ha='center')
        self.ax.text(15000, 0.001, 'WHITE\nDWARFS', color='#aaaaff',
                    fontsize=9, fontweight='bold', alpha=0.7, ha='center')

        # Spectral class labels at top
        spectral_temps = [35000, 20000, 9000, 7000, 5500, 4500, 3200]
        spectral_classes = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
        for T, cls in zip(spectral_temps, spectral_classes):
            self.ax.text(T, 5e6, cls, color='white', fontsize=11,
                        fontweight='bold', ha='center', va='bottom')

        # Axis labels
        self.ax.set_xlabel('Temperature (K)', color='white', fontsize=11)
        self.ax.set_ylabel('Luminosity (L☉)', color='white', fontsize=11)

        # Style
        self.ax.tick_params(colors='#888888', labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color('#444444')

        self.fig.canvas.draw_idle()

    def _update_info_panel(self):
        """Update the information panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        y = 0.95

        self.info_ax.text(0.5, y, 'STELLAR INFO', transform=self.info_ax.transAxes,
                         color='white', fontsize=12, fontweight='bold', ha='center')
        y -= 0.05

        if self.selected_star is not None:
            idx = self.selected_star
            name = self.star_names[idx]
            T = self.star_temps[idx]
            L = self.star_lums[idx]
            stype = self.star_types[idx]
            notes = self.star_notes[idx]

            color = temperature_to_rgb(T)
            R = luminosity_to_radius(L, T)

            y -= 0.03
            self.info_ax.text(0.5, y, name, transform=self.info_ax.transAxes,
                             color=color, fontsize=16, fontweight='bold', ha='center')
            y -= 0.04
            self.info_ax.text(0.5, y, stype, transform=self.info_ax.transAxes,
                             color='#888888', fontsize=11, ha='center')
            y -= 0.03
            self.info_ax.text(0.5, y, notes, transform=self.info_ax.transAxes,
                             color='#666666', fontsize=9, ha='center', style='italic')
            y -= 0.06

            self.info_ax.text(0.5, y, '─' * 20, transform=self.info_ax.transAxes,
                             color='#333333', ha='center')
            y -= 0.05

            # Properties
            props = [
                ('Temperature', f'{T:,.0f} K'),
                ('Luminosity', f'{L:,.2g} L☉'),
                ('Radius', f'{R:.2g} R☉'),
                ('Spectral Class', stype),
            ]

            for prop_name, prop_val in props:
                self.info_ax.text(0.1, y, prop_name + ':', transform=self.info_ax.transAxes,
                                 color='#888888', fontsize=10)
                self.info_ax.text(0.95, y, prop_val, transform=self.info_ax.transAxes,
                                 color='white', fontsize=10, ha='right')
                y -= 0.045

        else:
            y -= 0.05
            self.info_ax.text(0.5, y, 'Click on a star\nto see its properties',
                             transform=self.info_ax.transAxes,
                             color='#666666', fontsize=10, ha='center')

        # Legend
        y = 0.3
        self.info_ax.text(0.5, y, '─' * 20, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.04

        self.info_ax.text(0.5, y, 'KEYBOARD', transform=self.info_ax.transAxes,
                         color='#666666', fontsize=9, ha='center')
        y -= 0.04

        controls = [
            ('1', 'Highlight Main Sequence'),
            ('2', 'Highlight Giants'),
            ('3', 'Highlight Supergiants'),
            ('4', 'Highlight White Dwarfs'),
            ('5', 'Show All'),
            ('R', 'Reset'),
        ]

        for key, desc in controls:
            self.info_ax.text(0.1, y, key, transform=self.info_ax.transAxes,
                             color='#ffff88', fontsize=9, fontweight='bold')
            self.info_ax.text(0.2, y, desc, transform=self.info_ax.transAxes,
                             color='#666666', fontsize=8)
            y -= 0.035

        self.fig.canvas.draw_idle()

    def run(self):
        """Display the diagram."""
        self._update_info_panel()
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  HERTZSPRUNG-RUSSELL DIAGRAM")
    print("=" * 55)
    print()
    print("  The astronomer's most important diagram!")
    print("  Plotting stars by temperature and luminosity")
    print("  reveals patterns of stellar structure and evolution.")
    print()
    print("  Controls:")
    print("    CLICK  - Select a star")
    print("    1-4    - Highlight stellar types")
    print("    5      - Show all")
    print("    R      - Reset")
    print("    Q/ESC  - Quit")
    print()
    print("=" * 55)

    diagram = HRDiagram()
    diagram.run()


if __name__ == '__main__':
    main()
