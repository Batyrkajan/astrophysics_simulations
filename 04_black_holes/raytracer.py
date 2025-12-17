"""
Black Hole Ray Tracer
=====================

Visualize gravitational lensing around a Schwarzschild black hole.
Watch how light bends as it passes near the event horizon, creating
stunning visual effects like Einstein rings and the photon sphere.

Physics:
- Schwarzschild metric: ds² = -(1-rs/r)dt² + (1-rs/r)⁻¹dr² + r²dΩ²
- Null geodesics for photon paths
- Photon sphere at r = 1.5 rs
- Event horizon at r = rs (Schwarzschild radius)

Features:
- Real-time ray tracing
- Background starfield distortion
- Accretion disk (optional)
- Einstein ring visualization
- Adjustable observer distance

Controls:
    ←/→     - Rotate view
    ↑/↓     - Move closer/further
    A       - Toggle accretion disk
    G       - Toggle grid background
    R       - Reset view
    Q/ESC   - Quit

Run:
    python raytracer.py
"""

import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


# =============================================================================
# BLACK HOLE PHYSICS
# =============================================================================

class SchwarzschildBlackHole:
    """
    Schwarzschild (non-rotating) black hole.

    In geometric units where G = c = 1:
    - Schwarzschild radius rs = 2M
    - Photon sphere at r = 3M = 1.5 rs
    - Event horizon at r = 2M = rs
    """

    def __init__(self, mass=1.0):
        """
        Initialize black hole.

        Args:
            mass: Mass in geometric units (rs = 2*mass)
        """
        self.mass = mass
        self.rs = 2 * mass  # Schwarzschild radius
        self.r_photon = 1.5 * self.rs  # Photon sphere

    def metric_coefficient(self, r):
        """
        Metric coefficient g_tt = -(1 - rs/r)

        Also equals 1/g_rr
        """
        return 1 - self.rs / r

    def effective_potential(self, r, L):
        """
        Effective potential for photon with angular momentum L.

        V_eff = (1 - rs/r)(L²/r²)
        """
        return (1 - self.rs/r) * L**2 / r**2

    def geodesic_equations(self, state, phi, b):
        """
        Geodesic equations for a photon in the equatorial plane.

        We use phi as the affine parameter instead of proper time.
        State = [r, dr/dphi]

        For a photon: b = L/E is the impact parameter

        The equation of motion is:
        (dr/dphi)² = r⁴/b² - r²(1 - rs/r)

        Differentiate to get:
        d²r/dphi² = 2r³/b² - r(1 - rs/r) - rs*r/2
        """
        r, dr_dphi = state

        if r < self.rs * 1.01:  # Fell into black hole
            return [0, 0]

        # Second derivative
        d2r_dphi2 = (2 * r**3 / b**2 - r * (1 - self.rs/r)
                    - self.rs * r / 2 + self.rs * r / (2*(1 - self.rs/r)))

        return [dr_dphi, d2r_dphi2]

    def trace_ray(self, r0, theta0, n_points=500, max_phi=4*np.pi):
        """
        Trace a ray from the observer.

        Args:
            r0: Starting radius (observer distance)
            theta0: Initial angle from radial direction
            n_points: Number of integration points
            max_phi: Maximum azimuthal angle to trace

        Returns:
            Array of (r, phi) points along the geodesic
        """
        # Impact parameter: b = r * sin(theta) for distant observer
        b = r0 * np.sin(theta0)

        if abs(b) < 1e-10:  # Radial trajectory
            return np.array([[r0, 0], [self.rs, 0]])

        # Critical impact parameter (photon sphere)
        b_crit = self.r_photon * np.sqrt(3)

        # Initial dr/dphi
        # From: (dr/dphi)² = r⁴/b² - r²(1 - rs/r)
        dr_dphi_sq = r0**4/b**2 - r0**2*(1 - self.rs/r0)

        if dr_dphi_sq < 0:
            return None  # Invalid trajectory

        # Direction: moving towards black hole initially if theta > 90°
        dr_dphi = -np.sqrt(dr_dphi_sq) * np.sign(np.cos(theta0))

        # Integrate
        phi = np.linspace(0, max_phi, n_points)
        try:
            solution = odeint(self.geodesic_equations, [r0, dr_dphi], phi, args=(b,))
        except:
            return None

        r = solution[:, 0]

        # Truncate at event horizon
        valid = r > self.rs * 1.01
        if not np.any(valid):
            return None

        last_valid = np.where(valid)[0][-1]
        return np.column_stack([r[:last_valid+1], phi[:last_valid+1]])


class RayTracer:
    """
    Ray tracer for black hole visualization.
    """

    def __init__(self, bh, observer_distance=20, resolution=150):
        """
        Initialize ray tracer.

        Args:
            bh: SchwarzschildBlackHole instance
            observer_distance: Distance from black hole (in rs units)
            resolution: Image resolution (pixels per side)
        """
        self.bh = bh
        self.observer_distance = observer_distance
        self.resolution = resolution

        # Background (stars or grid)
        self.background_type = 'stars'
        self.show_disk = False

        # Generate background
        self._generate_starfield()

    def _generate_starfield(self, n_stars=2000):
        """Generate random background star positions."""
        np.random.seed(42)
        self.star_phi = np.random.uniform(0, 2*np.pi, n_stars)
        self.star_theta = np.arccos(np.random.uniform(-1, 1, n_stars))
        self.star_brightness = np.random.uniform(0.3, 1.0, n_stars)

    def render_frame(self):
        """
        Render a single frame.

        Returns image array.
        """
        res = self.resolution
        image = np.zeros((res, res, 3))

        # Field of view
        fov = np.pi / 4  # 45 degrees

        # For each pixel, trace a ray
        for i in range(res):
            for j in range(res):
                # Pixel coordinates to angles
                x = (j - res/2) / res * fov
                y = (i - res/2) / res * fov

                # Angle from line of sight
                theta = np.sqrt(x**2 + y**2)
                if theta < 1e-10:
                    theta = 1e-10

                # Trace ray
                color = self._trace_pixel_ray(theta, np.arctan2(y, x))
                image[i, j] = color

        return image

    def _trace_pixel_ray(self, theta, phi_dir):
        """
        Trace a single ray and determine its color.

        Args:
            theta: Angle from line of sight
            phi_dir: Direction in image plane

        Returns:
            RGB color tuple
        """
        r0 = self.observer_distance
        bh = self.bh

        # Impact parameter
        b = r0 * np.sin(theta)

        # Critical impact parameter
        b_crit = 3 * np.sqrt(3) * bh.mass

        # Check if ray falls into black hole
        if abs(b) < b_crit:
            # Check more precisely
            trajectory = bh.trace_ray(r0, theta)
            if trajectory is None or np.min(trajectory[:, 0]) < bh.rs * 1.1:
                # Fell into black hole - black
                return (0, 0, 0)

        # Calculate deflection angle using weak-field approximation
        # δ ≈ 4M/b for b >> rs (Einstein angle)
        if abs(b) > bh.rs:
            deflection = 4 * bh.mass / b
        else:
            deflection = np.pi  # Strong deflection

        # Find background direction
        # The ray came from direction theta + deflection
        final_theta = theta + deflection
        final_phi = phi_dir

        # Accretion disk check
        if self.show_disk:
            # Simple disk model: flat disk in equatorial plane
            disk_inner = 3 * bh.rs
            disk_outer = 10 * bh.rs

            # Check if ray intersects disk (very approximate)
            if abs(deflection) > 0.1 and abs(b) < disk_outer and abs(b) > disk_inner:
                # Disk color based on radius (temperature)
                temp_factor = 1 - (abs(b) - disk_inner) / (disk_outer - disk_inner)
                return (1.0, 0.8 * temp_factor, 0.3 * temp_factor)

        # Background color
        if self.background_type == 'stars':
            return self._starfield_color(final_theta, final_phi)
        else:
            return self._grid_color(final_theta, final_phi)

    def _starfield_color(self, theta, phi):
        """Get color from starfield at given direction."""
        # Check if any star is near this direction
        for i in range(len(self.star_phi)):
            # Angular distance
            d_phi = abs(phi - self.star_phi[i])
            d_theta = abs(theta - self.star_theta[i])

            if d_phi > np.pi:
                d_phi = 2*np.pi - d_phi

            if d_theta < 0.05 and d_phi < 0.05:
                b = self.star_brightness[i]
                return (b, b, b * 0.9)

        # Dark background
        return (0.01, 0.01, 0.02)

    def _grid_color(self, theta, phi):
        """Get color from grid pattern."""
        # Create a grid pattern on celestial sphere
        grid_spacing = np.pi / 8

        phi_line = abs(phi % grid_spacing) < 0.03
        theta_line = abs(theta % grid_spacing) < 0.03

        if phi_line or theta_line:
            return (0.3, 0.3, 0.4)
        else:
            return (0.05, 0.05, 0.08)


def fast_render(bh, observer_distance, resolution=200, show_disk=False):
    """
    Fast rendering using vectorized operations.

    This is an approximation for interactive visualization.
    Includes: black hole shadow, photon ring, Einstein ring,
    gravitationally lensed starfield, and accretion disk.
    """
    res = resolution
    fov = np.pi / 3

    # Create coordinate grids
    x = np.linspace(-fov/2, fov/2, res)
    y = np.linspace(-fov/2, fov/2, res)
    X, Y = np.meshgrid(x, y)

    # Angle from line of sight
    theta = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Y, X)

    # Impact parameter
    b = observer_distance * np.sin(np.clip(theta, 1e-10, np.pi/2))

    # Critical impact parameter (photon sphere capture)
    b_crit = 3 * np.sqrt(3) * bh.mass

    # Initialize image with dark space background
    image = np.zeros((res, res, 3))
    image[:, :] = [0.02, 0.02, 0.04]  # Very dark blue

    # Black hole shadow (rays captured by event horizon)
    shadow = np.abs(b) < b_crit * 0.92
    image[shadow] = [0, 0, 0]  # Pure black

    # Photon ring (light orbiting near photon sphere)
    # Multiple rings due to light orbiting multiple times
    ring1 = (np.abs(b) > b_crit * 0.92) & (np.abs(b) < b_crit * 0.97)
    ring2 = (np.abs(b) > b_crit * 0.97) & (np.abs(b) < b_crit * 1.02)
    ring3 = (np.abs(b) > b_crit * 1.02) & (np.abs(b) < b_crit * 1.08)

    image[ring1] = [1.0, 0.9, 0.7]   # Bright inner ring
    image[ring2] = [1.0, 0.7, 0.4]   # Middle ring
    image[ring3] = [0.8, 0.5, 0.3]   # Outer ring

    # Deflection for remaining rays
    outside = np.abs(b) >= b_crit * 1.08
    deflection = np.zeros_like(b)
    deflection[outside] = 4 * bh.mass / np.maximum(b[outside], 0.01)

    # Einstein ring effect (strong lensing region)
    einstein_ring = (deflection > 0.8) & (deflection < 2.0) & outside
    intensity = 1.0 - np.abs(deflection - 1.4) / 0.6
    intensity = np.clip(intensity, 0, 1)
    image[einstein_ring, 0] = 0.4 + 0.4 * intensity[einstein_ring]
    image[einstein_ring, 1] = 0.5 + 0.4 * intensity[einstein_ring]
    image[einstein_ring, 2] = 0.8 + 0.2 * intensity[einstein_ring]

    # Lensed background stars
    np.random.seed(42)
    star_field = np.random.random((res, res))

    # Stars appear brighter near the photon sphere due to lensing magnification
    magnification = 1.0 / (1.0 + deflection * 0.5)
    star_threshold = 0.015 * magnification

    stars = (star_field < star_threshold) & outside & ~einstein_ring
    star_brightness = np.random.random((res, res)) * 0.5 + 0.5
    image[stars, 0] = star_brightness[stars]
    image[stars, 1] = star_brightness[stars]
    image[stars, 2] = star_brightness[stars] * 0.95

    # Accretion disk with Doppler beaming
    if show_disk:
        disk_r = np.abs(b)
        disk_inner = 3 * bh.rs
        disk_outer = 12 * bh.rs

        # Disk is in equatorial plane - visible where |Y| is small
        disk_thickness = 0.3 + 0.2 * (disk_r / disk_outer)
        in_disk = (disk_r > disk_inner) & (disk_r < disk_outer) & (np.abs(Y) < disk_thickness)

        # Temperature profile: hotter near center
        temp = np.zeros_like(disk_r)
        temp[in_disk] = 1 - (disk_r[in_disk] - disk_inner) / (disk_outer - disk_inner)
        temp = np.clip(temp, 0, 1)

        # Doppler beaming: approaching side (X < 0) is brighter
        doppler = np.ones_like(X)
        doppler[in_disk] = 1.0 + 0.5 * np.tanh(-X[in_disk] / (disk_outer * 0.3))

        # Disk colors (hot gas: yellow-orange-red)
        image[in_disk, 0] = np.clip(temp[in_disk] * doppler[in_disk] * 1.2, 0, 1)
        image[in_disk, 1] = np.clip(temp[in_disk] * doppler[in_disk] * 0.6, 0, 1)
        image[in_disk, 2] = np.clip(temp[in_disk] * doppler[in_disk] * 0.15, 0, 1)

        # Add glow around the disk
        glow_region = (disk_r > disk_inner * 0.8) & (disk_r < disk_outer * 1.2) & ~in_disk
        glow_region = glow_region & (np.abs(Y) < disk_thickness * 2)
        glow_intensity = np.exp(-np.abs(Y) / disk_thickness) * 0.3
        image[glow_region, 0] += glow_intensity[glow_region] * 0.8
        image[glow_region, 1] += glow_intensity[glow_region] * 0.3
        image[glow_region, 2] += glow_intensity[glow_region] * 0.1
        image = np.clip(image, 0, 1)

    return image


# =============================================================================
# VISUALIZATION
# =============================================================================

class BlackHoleVisualizer:
    """
    Interactive black hole visualization.
    """

    def __init__(self):
        self.bh = SchwarzschildBlackHole(mass=1.0)
        self.observer_distance = 15
        self.view_angle = 0
        self.show_disk = True
        self.show_grid = False
        self.resolution = 200

        self._setup_figure()
        self._setup_controls()
        self._render()

    def _setup_figure(self):
        """Create figure."""
        self.fig = plt.figure(figsize=(12, 10), facecolor='black')

        # Main image
        self.ax = self.fig.add_axes([0.05, 0.15, 0.7, 0.8])
        self.ax.set_facecolor('black')
        self.ax.axis('off')

        # Info panel
        self.info_ax = self.fig.add_axes([0.78, 0.15, 0.2, 0.8])
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        # Title
        self.fig.suptitle('BLACK HOLE RAY TRACER', color='white',
                         fontsize=16, fontweight='bold', y=0.97)

        # Controls
        self.fig.text(0.4, 0.05, '←→: Rotate   ↑↓: Distance   A: Disk   G: Grid   R: Reset',
                     color='#666666', fontsize=9, ha='center')

        # Image display
        self.im = self.ax.imshow(np.zeros((self.resolution, self.resolution, 3)),
                                 extent=[-1, 1, -1, 1], origin='lower')

    def _setup_controls(self):
        """Setup keyboard controls."""
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == 'left':
            self.view_angle -= 0.1
        elif event.key == 'right':
            self.view_angle += 0.1
        elif event.key == 'up':
            self.observer_distance = max(5, self.observer_distance - 1)
        elif event.key == 'down':
            self.observer_distance = min(50, self.observer_distance + 1)
        elif event.key in ['a', 'A']:
            self.show_disk = not self.show_disk
        elif event.key in ['g', 'G']:
            self.show_grid = not self.show_grid
        elif event.key in ['r', 'R']:
            self.observer_distance = 15
            self.view_angle = 0
            self.show_disk = True
        elif event.key in ['q', 'Q', 'escape']:
            plt.close(self.fig)
            return

        self._render()

    def _render(self):
        """Render the current view."""
        image = fast_render(self.bh, self.observer_distance,
                           self.resolution, self.show_disk)

        # Apply view rotation (simple roll)
        if abs(self.view_angle) > 0.01:
            from scipy.ndimage import rotate
            image = rotate(image, np.degrees(self.view_angle), reshape=False)

        self.im.set_data(image)
        self._update_info()
        self.fig.canvas.draw_idle()

    def _update_info(self):
        """Update info panel."""
        self.info_ax.clear()
        self.info_ax.set_facecolor('#0a0a15')
        self.info_ax.axis('off')

        y = 0.95

        self.info_ax.text(0.5, y, 'BLACK HOLE', transform=self.info_ax.transAxes,
                         color='white', fontsize=12, fontweight='bold', ha='center')
        y -= 0.05
        self.info_ax.text(0.5, y, 'Schwarzschild', transform=self.info_ax.transAxes,
                         color='#888888', fontsize=9, ha='center')
        y -= 0.08

        self.info_ax.text(0.5, y, '─' * 15, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.06

        # Properties
        rs = self.bh.rs
        props = [
            ('Mass', f'{self.bh.mass:.1f} M'),
            ('Schwarzschild R', f'{rs:.1f} Rs'),
            ('Photon Sphere', f'{1.5*rs:.1f} Rs'),
            ('Observer Dist', f'{self.observer_distance:.0f} Rs'),
            ('', ''),
            ('Accretion Disk', 'ON' if self.show_disk else 'OFF'),
        ]

        for name, val in props:
            if name:
                self.info_ax.text(0.05, y, name + ':', transform=self.info_ax.transAxes,
                                 color='#888888', fontsize=9)
                self.info_ax.text(0.95, y, str(val), transform=self.info_ax.transAxes,
                                 color='white', fontsize=9, ha='right')
            y -= 0.045

        # Visual elements legend
        y -= 0.05
        self.info_ax.text(0.5, y, '─' * 15, transform=self.info_ax.transAxes,
                         color='#333333', ha='center')
        y -= 0.05

        self.info_ax.text(0.5, y, 'VISUAL ELEMENTS', transform=self.info_ax.transAxes,
                         color='#666666', fontsize=9, ha='center')
        y -= 0.05

        elements = [
            ('● Black region', 'Event horizon', 'black'),
            ('● Orange ring', 'Photon sphere', '#ff8844'),
            ('● Blue ring', 'Einstein ring', '#6688ff'),
            ('● Orange glow', 'Accretion disk', '#ff6622'),
        ]

        for marker, desc, color in elements:
            self.info_ax.text(0.05, y, marker, transform=self.info_ax.transAxes,
                             color=color, fontsize=9)
            self.info_ax.text(0.25, y, desc, transform=self.info_ax.transAxes,
                             color='#888888', fontsize=8)
            y -= 0.04

        self.fig.canvas.draw_idle()

    def run(self):
        """Start visualization."""
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 55)
    print("  BLACK HOLE RAY TRACER")
    print("=" * 55)
    print()
    print("  Visualizing gravitational lensing around a")
    print("  Schwarzschild (non-rotating) black hole.")
    print()
    print("  Controls:")
    print("    ←/→     - Rotate view")
    print("    ↑/↓     - Move closer/further")
    print("    A       - Toggle accretion disk")
    print("    G       - Toggle grid")
    print("    R       - Reset")
    print("    Q/ESC   - Quit")
    print()
    print("  Features visible:")
    print("    - Event horizon (black shadow)")
    print("    - Photon sphere (bright ring)")
    print("    - Einstein ring (gravitational lensing)")
    print("    - Accretion disk (hot gas)")
    print()
    print("=" * 55)

    viz = BlackHoleVisualizer()
    viz.run()


if __name__ == '__main__':
    main()
