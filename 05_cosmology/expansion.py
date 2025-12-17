"""
Universe Expansion Simulator
============================

Visualize the expansion of the universe using the Friedmann equations.
Explore different cosmological models and see how the universe evolves
from the Big Bang to its ultimate fate.

Physics:
- Friedmann equations from General Relativity
- Scale factor a(t) describes universe expansion
- Hubble parameter H(t) = ȧ/a
- Different components: radiation, matter, dark energy

Cosmological Models:
- ΛCDM: Current best model (matter + dark energy)
- Einstein-de Sitter: Matter-only universe
- de Sitter: Dark energy dominated
- Open/Closed: Different spatial curvatures

Controls:
    SPACE     - Pause/Resume
    1-4       - Select cosmological model
    ↑/↓       - Adjust Ω_Λ (dark energy)
    ←/→       - Adjust Ω_m (matter)
    R         - Reset to ΛCDM
    Q/ESC     - Quit

Run:
    python expansion.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint, quad

from shared.constants import YEAR, c, G


# =============================================================================
# COSMOLOGY PHYSICS
# =============================================================================

# Cosmological constants (current values)
H0_SI = 70.0  # km/s/Mpc in SI units
H0 = 70.0 / 3.086e19  # Convert to 1/s
t_H = 1 / H0  # Hubble time in seconds

# Convert to Gyr for convenience
GYR = 1e9 * 365.25 * 24 * 3600
t_H_Gyr = t_H / GYR  # ~14 Gyr


class FriedmannUniverse:
    """
    Solve the Friedmann equations for universe evolution.

    The Friedmann equation:
    (ȧ/a)² = H₀² [Ω_r/a⁴ + Ω_m/a³ + Ω_k/a² + Ω_Λ]

    Where:
    - a = scale factor (a=1 today)
    - Ω_r = radiation density
    - Ω_m = matter density
    - Ω_k = curvature term
    - Ω_Λ = dark energy density
    """

    # Predefined models
    MODELS = {
        'ΛCDM': {'Omega_m': 0.3, 'Omega_L': 0.7, 'Omega_r': 9e-5, 'name': 'ΛCDM (Standard Model)'},
        'EdS': {'Omega_m': 1.0, 'Omega_L': 0.0, 'Omega_r': 0.0, 'name': 'Einstein-de Sitter'},
        'deSitter': {'Omega_m': 0.0, 'Omega_L': 1.0, 'Omega_r': 0.0, 'name': 'de Sitter (Dark Energy)'},
        'Open': {'Omega_m': 0.3, 'Omega_L': 0.0, 'Omega_r': 0.0, 'name': 'Open Universe'},
    }

    def __init__(self, Omega_m=0.3, Omega_L=0.7, Omega_r=9e-5, H0=70.0):
        """
        Initialize cosmological model.

        Args:
            Omega_m: Matter density parameter
            Omega_L: Dark energy density parameter
            Omega_r: Radiation density parameter
            H0: Hubble constant in km/s/Mpc
        """
        self.Omega_m = Omega_m
        self.Omega_L = Omega_L
        self.Omega_r = Omega_r
        self.H0 = H0

        # Curvature from constraint Ω_total = 1 for flat universe
        self.Omega_k = 1.0 - Omega_m - Omega_L - Omega_r

        # Hubble time
        self.t_H = 13.97  # Gyr, for H0=70

    def E(self, a):
        """
        Dimensionless Hubble parameter E(a) = H(a)/H0

        E² = Ω_r/a⁴ + Ω_m/a³ + Ω_k/a² + Ω_Λ
        """
        return np.sqrt(self.Omega_r / a**4 +
                       self.Omega_m / a**3 +
                       self.Omega_k / a**2 +
                       self.Omega_L)

    def H(self, a):
        """Hubble parameter H(a) in 1/Gyr"""
        return self.H0 / 978.0 * self.E(a)  # Convert km/s/Mpc to 1/Gyr

    def da_dt(self, a, t):
        """
        Time derivative of scale factor.

        da/dt = a * H(a)
        """
        if a < 1e-10:
            return 1e-10
        return a * self.H(a)

    def lookback_time_integrand(self, a):
        """Integrand for lookback time calculation."""
        if a < 1e-10:
            return 0
        return 1.0 / (a * self.E(a))

    def age_of_universe(self, a=1.0):
        """
        Age of universe when scale factor is a.

        t = (1/H0) ∫₀^a da'/(a' * E(a'))
        """
        if a < 1e-10:
            return 0

        result, _ = quad(self.lookback_time_integrand, 1e-10, a)
        return self.t_H * result

    def evolve(self, t_max=30.0, n_points=1000):
        """
        Evolve the universe from Big Bang to t_max Gyr.

        Returns:
            t: time array in Gyr
            a: scale factor array
        """
        t = np.linspace(0.001, t_max, n_points)

        # Solve ODE
        a0 = 1e-6  # Start near Big Bang
        solution = odeint(self.da_dt, a0, t)
        a = solution[:, 0]

        return t, a

    def fate(self):
        """Determine the ultimate fate of this universe."""
        if self.Omega_L > 0:
            if self.Omega_m + self.Omega_L > 1.01:
                return "Big Crunch (eventually)"
            else:
                return "Eternal expansion (exponential)"
        elif self.Omega_m > 1:
            return "Big Crunch"
        elif self.Omega_m < 1:
            return "Eternal expansion (slowing)"
        else:
            return "Eternal expansion (critical)"


# =============================================================================
# VISUALIZATION
# =============================================================================

class CosmologyVisualizer:
    """
    Interactive cosmology visualization.
    """

    def __init__(self):
        self.universe = FriedmannUniverse()
        self.current_model = 'ΛCDM'
        self.running = False
        self.current_time = 0  # Current animation time in Gyr
        self.time_index = 0

        # Pre-compute evolution
        self._compute_evolution()

        self._setup_figure()
        self._setup_controls()
        self._update_display()

    def _compute_evolution(self):
        """Pre-compute universe evolution."""
        self.t, self.a = self.universe.evolve(t_max=50.0, n_points=500)

        # Find age of universe today (a=1)
        today_idx = np.argmin(np.abs(self.a - 1.0))
        self.today_time = self.t[today_idx]

    def _setup_figure(self):
        """Create figure."""
        self.fig = plt.figure(figsize=(16, 9), facecolor='black')

        # Universe visualization (expanding dots)
        self.uni_ax = self.fig.add_axes([0.02, 0.1, 0.45, 0.8])
        self.uni_ax.set_facecolor('black')
        self.uni_ax.set_aspect('equal')
        self.uni_ax.axis('off')

        # Scale factor plot
        self.scale_ax = self.fig.add_axes([0.52, 0.55, 0.3, 0.35])
        self.scale_ax.set_facecolor('#0a0a15')

        # Hubble parameter plot
        self.hubble_ax = self.fig.add_axes([0.52, 0.1, 0.3, 0.35])
        self.hubble_ax.set_facecolor('#0a0a15')

        # Info panel
        self.info_ax = self.fig.add_axes([0.84, 0.1, 0.15, 0.8])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('UNIVERSE EXPANSION SIMULATOR',
                         color='white', fontsize=16, fontweight='bold', y=0.97)

        # Controls
        self.fig.text(0.35, 0.02,
                     'SPACE: Play  1-4: Models  ↑↓←→: Adjust Ω  R: Reset',
                     color='#666666', fontsize=9, ha='center')

        # Generate random "galaxies" for visualization
        np.random.seed(42)
        n_galaxies = 100
        self.galaxy_angles = np.random.uniform(0, 2*np.pi, n_galaxies)
        self.galaxy_radii = np.random.uniform(0.1, 0.9, n_galaxies)
        self.galaxy_colors = plt.cm.Spectral(np.random.random(n_galaxies))

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.running = not self.running
        elif event.key == '1':
            self._set_model('ΛCDM')
        elif event.key == '2':
            self._set_model('EdS')
        elif event.key == '3':
            self._set_model('deSitter')
        elif event.key == '4':
            self._set_model('Open')
        elif event.key == 'up':
            self.universe.Omega_L = min(1.5, self.universe.Omega_L + 0.05)
            self._compute_evolution()
            self._update_display()
        elif event.key == 'down':
            self.universe.Omega_L = max(0, self.universe.Omega_L - 0.05)
            self._compute_evolution()
            self._update_display()
        elif event.key == 'right':
            self.universe.Omega_m = min(2.0, self.universe.Omega_m + 0.05)
            self._compute_evolution()
            self._update_display()
        elif event.key == 'left':
            self.universe.Omega_m = max(0, self.universe.Omega_m - 0.05)
            self._compute_evolution()
            self._update_display()
        elif event.key in ['r', 'R']:
            self._set_model('ΛCDM')
            self.time_index = 0
            self.running = False
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)

    def _set_model(self, name):
        """Set a predefined cosmological model."""
        if name in FriedmannUniverse.MODELS:
            params = FriedmannUniverse.MODELS[name]
            self.universe = FriedmannUniverse(
                Omega_m=params['Omega_m'],
                Omega_L=params['Omega_L'],
                Omega_r=params['Omega_r']
            )
            self.current_model = name
            self._compute_evolution()
            self.time_index = 0
            self._update_display()

    def _draw_universe(self):
        """Draw the expanding universe visualization."""
        self.uni_ax.clear()
        self.uni_ax.set_facecolor('black')
        self.uni_ax.axis('off')
        self.uni_ax.set_xlim(-1.5, 1.5)
        self.uni_ax.set_ylim(-1.5, 1.5)

        # Current scale factor
        if self.time_index < len(self.a):
            scale = self.a[self.time_index]
            time = self.t[self.time_index]
        else:
            scale = self.a[-1]
            time = self.t[-1]

        # Normalize scale for visualization
        vis_scale = min(scale, 3.0)

        # Draw galaxies at scaled positions
        for i in range(len(self.galaxy_angles)):
            r = self.galaxy_radii[i] * vis_scale
            if r < 1.4:  # Only draw visible galaxies
                x = r * np.cos(self.galaxy_angles[i])
                y = r * np.sin(self.galaxy_angles[i])

                # Size based on distance (larger = closer = bigger)
                size = max(5, 30 - 20 * self.galaxy_radii[i])

                self.uni_ax.plot(x, y, 'o', color=self.galaxy_colors[i],
                               markersize=size, alpha=0.8)

        # Observer at center
        self.uni_ax.plot(0, 0, '*', color='yellow', markersize=15, zorder=10)
        self.uni_ax.text(0, -0.15, 'Observer', color='yellow', ha='center', fontsize=8)

        # Circle showing horizon
        theta = np.linspace(0, 2*np.pi, 100)
        self.uni_ax.plot(1.3*np.cos(theta), 1.3*np.sin(theta), '--',
                        color='#444444', linewidth=1)

        # Time label
        self.uni_ax.text(0, 1.4, f't = {time:.2f} Gyr', color='white',
                        ha='center', fontsize=12, fontweight='bold')

        if time < self.today_time:
            era = "PAST"
            color = '#ff8888'
        elif abs(time - self.today_time) < 0.5:
            era = "TODAY"
            color = '#88ff88'
        else:
            era = "FUTURE"
            color = '#8888ff'

        self.uni_ax.text(0, 1.25, era, color=color, ha='center', fontsize=10)

    def _draw_scale_factor(self):
        """Draw scale factor evolution."""
        self.scale_ax.clear()
        self.scale_ax.set_facecolor('#0a0a15')

        # Full evolution
        self.scale_ax.plot(self.t, self.a, '-', color='#4488ff', linewidth=2, label='a(t)')

        # Current point
        if self.time_index < len(self.t):
            self.scale_ax.plot(self.t[self.time_index], self.a[self.time_index],
                              'o', color='white', markersize=10, zorder=10)

        # Today line
        self.scale_ax.axhline(y=1, color='#88ff88', linestyle='--', alpha=0.5, label='Today (a=1)')
        self.scale_ax.axvline(x=self.today_time, color='#88ff88', linestyle='--', alpha=0.5)

        self.scale_ax.set_xlabel('Time (Gyr)', color='white', fontsize=9)
        self.scale_ax.set_ylabel('Scale Factor a(t)', color='white', fontsize=9)
        self.scale_ax.set_title('Universe Scale Factor', color='white', fontsize=10)
        self.scale_ax.tick_params(colors='#888888', labelsize=8)
        self.scale_ax.set_ylim(0, max(3, np.max(self.a) * 1.1))
        self.scale_ax.legend(loc='upper left', fontsize=7)

    def _draw_hubble(self):
        """Draw Hubble parameter evolution."""
        self.hubble_ax.clear()
        self.hubble_ax.set_facecolor('#0a0a15')

        # Calculate H(a) for the evolution
        H = np.array([self.universe.H(a) for a in self.a])

        self.hubble_ax.plot(self.t, H, '-', color='#ff8844', linewidth=2)

        # Current point
        if self.time_index < len(self.t):
            self.hubble_ax.plot(self.t[self.time_index], H[self.time_index],
                               'o', color='white', markersize=10, zorder=10)

        # Today
        self.hubble_ax.axvline(x=self.today_time, color='#88ff88', linestyle='--', alpha=0.5)

        self.hubble_ax.set_xlabel('Time (Gyr)', color='white', fontsize=9)
        self.hubble_ax.set_ylabel('H(t) [1/Gyr]', color='white', fontsize=9)
        self.hubble_ax.set_title('Hubble Parameter', color='white', fontsize=10)
        self.hubble_ax.tick_params(colors='#888888', labelsize=8)
        self.hubble_ax.set_yscale('log')
        self.hubble_ax.set_ylim(0.01, max(H) * 2)

    def _update_info_panel(self):
        """Update info panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        y = 0.95

        self.info_ax.text(0.5, y, 'MODEL', transform=self.info_ax.transAxes,
                         color='white', fontsize=11, fontweight='bold', ha='center')
        y -= 0.04

        # Model name
        model_name = FriedmannUniverse.MODELS.get(self.current_model, {}).get('name', 'Custom')
        self.info_ax.text(0.5, y, model_name, transform=self.info_ax.transAxes,
                         color='#88aaff', fontsize=9, ha='center')
        y -= 0.06

        self.info_ax.text(0.5, y, '─' * 12, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.05

        # Parameters
        params = [
            ('Ω_m (matter)', f'{self.universe.Omega_m:.3f}'),
            ('Ω_Λ (dark E)', f'{self.universe.Omega_L:.3f}'),
            ('Ω_r (radiation)', f'{self.universe.Omega_r:.1e}'),
            ('Ω_k (curvature)', f'{self.universe.Omega_k:.3f}'),
            ('', ''),
            ('H₀', f'{self.universe.H0:.1f} km/s/Mpc'),
            ('Age today', f'{self.today_time:.1f} Gyr'),
        ]

        for name, val in params:
            if name:
                self.info_ax.text(0.05, y, name, transform=self.info_ax.transAxes,
                                 color='#888888', fontsize=8)
                self.info_ax.text(0.95, y, val, transform=self.info_ax.transAxes,
                                 color='white', fontsize=8, ha='right')
            y -= 0.04

        # Fate
        y -= 0.03
        self.info_ax.text(0.5, y, '─' * 12, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.04

        self.info_ax.text(0.5, y, 'FATE', transform=self.info_ax.transAxes,
                         color='#666666', fontsize=9, ha='center')
        y -= 0.04

        fate = self.universe.fate()
        self.info_ax.text(0.5, y, fate, transform=self.info_ax.transAxes,
                         color='#ffaa44', fontsize=8, ha='center', wrap=True)

    def _update_display(self):
        """Update all display elements."""
        self._draw_universe()
        self._draw_scale_factor()
        self._draw_hubble()
        self._update_info_panel()
        self.fig.canvas.draw_idle()

    def animate(self, frame):
        """Animation step."""
        if self.running:
            self.time_index += 1
            if self.time_index >= len(self.t):
                self.time_index = 0  # Loop

            self._update_display()

        return []

    def run(self):
        """Start visualization."""
        self.anim = FuncAnimation(self.fig, self.animate, frames=None,
                                  interval=100, blit=False, cache_frame_data=False)
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  UNIVERSE EXPANSION SIMULATOR")
    print("=" * 55)
    print()
    print("  Watch the universe expand from the Big Bang!")
    print()
    print("  Controls:")
    print("    SPACE   - Play/Pause animation")
    print("    1       - ΛCDM model (standard cosmology)")
    print("    2       - Einstein-de Sitter (matter only)")
    print("    3       - de Sitter (dark energy only)")
    print("    4       - Open universe")
    print("    ↑/↓     - Adjust dark energy (Ω_Λ)")
    print("    ←/→     - Adjust matter (Ω_m)")
    print("    R       - Reset")
    print("    Q/ESC   - Quit")
    print()
    print("  Current best values (ΛCDM):")
    print("    Ω_m = 0.3 (matter)")
    print("    Ω_Λ = 0.7 (dark energy)")
    print()
    print("=" * 55)

    viz = CosmologyVisualizer()
    viz.run()


if __name__ == '__main__':
    main()
