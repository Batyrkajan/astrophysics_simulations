"""
Shared Visualization Utilities
==============================

Common visualization functions and classes used across all simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as pe


# =============================================================================
# COLOR UTILITIES
# =============================================================================

def temperature_to_rgb(T):
    """
    Convert temperature to approximate stellar color.

    Based on blackbody radiation, approximates what color a
    star appears to human eyes.

    Args:
        T: Temperature in Kelvin (1000-50000K range)

    Returns:
        (r, g, b) tuple with values 0-1
    """
    T = np.clip(T, 1000, 50000)

    if T < 6600:
        r = 1.0
        g = np.clip(0.39 * np.log(T/100) - 0.634, 0, 1)
        if T < 2000:
            b = 0.0
        else:
            b = np.clip(0.543 * np.log(max(T/100 - 10, 1)) - 1.186, 0, 1)
    else:
        r = np.clip(1.29 * (T/100 - 60)**(-0.133), 0, 1)
        g = np.clip(1.13 * (T/100 - 60)**(-0.076), 0, 1)
        b = 1.0

    return (r, g, b)


def velocity_to_color(velocity, v_max=None):
    """
    Convert velocity magnitude to color (blue=slow, red=fast).

    Args:
        velocity: Velocity magnitude or array
        v_max: Maximum velocity for normalization

    Returns:
        RGB color(s)
    """
    if v_max is None:
        v_max = np.max(np.abs(velocity)) if np.any(velocity) else 1.0

    norm_v = np.clip(np.abs(velocity) / v_max, 0, 1)

    # Blue to white to red colormap
    r = norm_v
    g = 1 - np.abs(norm_v - 0.5) * 2
    b = 1 - norm_v

    if np.isscalar(velocity):
        return (r, g, b)
    return np.column_stack([r, g, b])


def create_colormap(name='space'):
    """
    Create custom colormaps for astrophysics visualizations.

    Available maps:
    - 'space': Dark blue to purple to white
    - 'heat': Black to red to yellow to white (temperature)
    - 'galaxy': Blue to white (for galaxy visualization)
    """
    if name == 'space':
        colors = ['#000011', '#110033', '#330066', '#6600aa', '#aa66ff', '#ffffff']
        return LinearSegmentedColormap.from_list('space', colors)

    elif name == 'heat':
        colors = ['#000000', '#330000', '#660000', '#aa0000',
                 '#ff0000', '#ff6600', '#ffaa00', '#ffff00', '#ffffff']
        return LinearSegmentedColormap.from_list('heat', colors)

    elif name == 'galaxy':
        colors = ['#000005', '#000020', '#000050', '#0000aa',
                 '#4444ff', '#8888ff', '#aaaaff', '#ffffff']
        return LinearSegmentedColormap.from_list('galaxy', colors)

    else:
        return plt.cm.viridis


# =============================================================================
# FIGURE SETUP
# =============================================================================

def setup_dark_figure(figsize=(12, 10), title=None):
    """
    Create a figure with dark theme styling.

    Args:
        figsize: Figure size tuple
        title: Optional title string

    Returns:
        fig, ax tuple
    """
    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0a0a15')

    # Style axes
    ax.tick_params(colors='#666666', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333333')

    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    if title:
        ax.set_title(title, color='white', fontsize=14, fontweight='bold')

    return fig, ax


def add_info_panel(fig, rect=[0.78, 0.1, 0.2, 0.8]):
    """
    Add an info panel to a figure.

    Args:
        fig: matplotlib figure
        rect: [left, bottom, width, height] in figure coords

    Returns:
        info_ax axes object
    """
    info_ax = fig.add_axes(rect)
    info_ax.set_facecolor('#0a0a15')
    info_ax.axis('off')
    return info_ax


# =============================================================================
# DRAWING UTILITIES
# =============================================================================

def draw_orbit(ax, radius, center=(0, 0), color='white', linestyle='--',
               alpha=0.5, linewidth=1, label=None):
    """
    Draw a circular orbit.

    Args:
        ax: matplotlib axes
        radius: Orbit radius
        center: Center point (x, y)
        color: Line color
        linestyle: Line style
        alpha: Transparency
        linewidth: Line width
        label: Optional label
    """
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color=color, linestyle=linestyle, alpha=alpha,
            linewidth=linewidth, label=label)


def draw_body(ax, position, radius, color='white', glow=True,
              label=None, label_offset=(0.1, 0.1)):
    """
    Draw a celestial body with optional glow effect.

    Args:
        ax: matplotlib axes
        position: (x, y) position
        radius: Display radius
        color: Body color
        glow: Whether to add glow effect
        label: Optional label text
        label_offset: (dx, dy) label offset
    """
    x, y = position

    if glow:
        # Glow layers
        for r_mult, alpha in [(1.5, 0.1), (1.3, 0.2), (1.1, 0.3)]:
            glow_circle = Circle((x, y), radius * r_mult,
                                color=color, alpha=alpha, zorder=5)
            ax.add_patch(glow_circle)

    # Main body
    body = Circle((x, y), radius, color=color, zorder=10)
    ax.add_patch(body)

    if label:
        ax.text(x + label_offset[0], y + label_offset[1], label,
               color=color, fontsize=9, ha='left', va='bottom',
               path_effects=[pe.withStroke(linewidth=2, foreground='black')])


def draw_vector(ax, start, direction, length=1, color='white',
                linewidth=2, label=None):
    """
    Draw a vector arrow.

    Args:
        ax: matplotlib axes
        start: (x, y) starting point
        direction: (dx, dy) direction (will be normalized)
        length: Arrow length
        color: Arrow color
        linewidth: Line width
        label: Optional label
    """
    dir_norm = np.array(direction)
    if np.linalg.norm(dir_norm) > 0:
        dir_norm = dir_norm / np.linalg.norm(dir_norm)

    end = np.array(start) + dir_norm * length

    arrow = FancyArrowPatch(start, end, arrowstyle='->', color=color,
                           linewidth=linewidth, mutation_scale=15)
    ax.add_patch(arrow)

    if label:
        mid = (np.array(start) + end) / 2
        ax.text(mid[0], mid[1], label, color=color, fontsize=9)


def add_starfield(ax, n_stars=200, extent=None, seed=42):
    """
    Add random background stars to an axes.

    Args:
        ax: matplotlib axes
        n_stars: Number of stars
        extent: (xmin, xmax, ymin, ymax) or None for auto
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)

    if extent is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        extent = (xlim[0], xlim[1], ylim[0], ylim[1])

    xmin, xmax, ymin, ymax = extent

    x = np.random.uniform(xmin, xmax, n_stars)
    y = np.random.uniform(ymin, ymax, n_stars)
    sizes = np.random.uniform(0.1, 1.0, n_stars)
    alphas = np.random.uniform(0.3, 0.8, n_stars)

    for i in range(n_stars):
        ax.plot(x[i], y[i], '.', color='white',
               markersize=sizes[i], alpha=alphas[i], zorder=1)


# =============================================================================
# ANIMATION HELPERS
# =============================================================================

def create_trail(ax, color='white', alpha=0.5, linewidth=1, max_length=200):
    """
    Create a trail line object for animations.

    Returns a dict with:
    - 'line': matplotlib line object
    - 'history': list to store position history
    - 'max_length': maximum trail length
    - 'update': function to update trail
    """
    line, = ax.plot([], [], '-', color=color, alpha=alpha,
                   linewidth=linewidth, zorder=5)
    history = []

    def update(position):
        history.append(np.array(position))
        if len(history) > max_length:
            history.pop(0)
        if history:
            arr = np.array(history)
            line.set_data(arr[:, 0], arr[:, 1])
        return line

    return {
        'line': line,
        'history': history,
        'max_length': max_length,
        'update': update
    }


# =============================================================================
# TEXT FORMATTING
# =============================================================================

def format_time(seconds):
    """
    Format time in human-readable units.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string
    """
    if seconds < 60:
        return f"{seconds:.1f} s"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hr"
    elif seconds < 31557600:  # 1 year
        return f"{seconds/86400:.1f} days"
    elif seconds < 31557600e3:  # 1000 years
        return f"{seconds/31557600:.1f} yr"
    elif seconds < 31557600e6:  # 1 Myr
        return f"{seconds/31557600e3:.1f} kyr"
    elif seconds < 31557600e9:  # 1 Gyr
        return f"{seconds/31557600e6:.1f} Myr"
    else:
        return f"{seconds/31557600e9:.1f} Gyr"


def format_distance(meters):
    """
    Format distance in human-readable units.

    Args:
        meters: Distance in meters

    Returns:
        Formatted string
    """
    AU = 1.496e11
    LY = 9.461e15
    PC = 3.086e16

    if meters < 1e6:
        return f"{meters:.0f} m"
    elif meters < 1e9:
        return f"{meters/1e3:.1f} km"
    elif meters < AU * 0.1:
        return f"{meters/1e6:.1f} Mm"
    elif meters < LY * 0.1:
        return f"{meters/AU:.2f} AU"
    elif meters < PC * 0.1:
        return f"{meters/LY:.2f} ly"
    elif meters < PC * 1e3:
        return f"{meters/PC:.2f} pc"
    elif meters < PC * 1e6:
        return f"{meters/PC/1e3:.2f} kpc"
    else:
        return f"{meters/PC/1e6:.2f} Mpc"


def format_mass(kg):
    """
    Format mass in human-readable units.

    Args:
        kg: Mass in kilograms

    Returns:
        Formatted string
    """
    M_SUN = 1.989e30
    M_EARTH = 5.972e24
    M_JUPITER = 1.898e27

    if kg < M_EARTH * 0.1:
        return f"{kg:.2e} kg"
    elif kg < M_JUPITER * 0.1:
        return f"{kg/M_EARTH:.2f} M⊕"
    elif kg < M_SUN * 0.1:
        return f"{kg/M_JUPITER:.2f} M♃"
    else:
        return f"{kg/M_SUN:.2f} M☉"
