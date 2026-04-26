"""Internal color assignment utilities. Not part of public API."""
from __future__ import annotations

import colorsys

# Six distinct primaries used first so common labels (Tumor/Stroma/Immune/...)
# always get the same easily-distinguished colors.
_PRIMARY_COLORS: list[list[int]] = [
    [255, 0, 0],      # red
    [0, 255, 0],      # green
    [0, 0, 255],      # blue
    [255, 255, 0],    # yellow
    [255, 0, 255],    # magenta
    [0, 255, 255],    # cyan
]

# Golden-ratio conjugate for hue stepping when n > 6.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _assign_colors(names: list[str]) -> dict[str, list[int]]:
    """Deterministically assign distinct RGB colors to a list of names.

    First 6 names get the primary colors (red/green/blue/yellow/magenta/cyan).
    Names beyond 6 use HSV hue stepping by the golden ratio conjugate so that
    successive colors are visually distinct.

    Parameters
    ----------
    names : list[str]
        Names to assign colors to. Order matters — same input produces same output.

    Returns
    -------
    dict[str, list[int]]
        Mapping from name to [R, G, B] with each channel in [0, 255].
    """
    out: dict[str, list[int]] = {}
    for i, name in enumerate(names):
        if i < len(_PRIMARY_COLORS):
            out[name] = list(_PRIMARY_COLORS[i])
        else:
            hue = ((i - len(_PRIMARY_COLORS) + 1) * _GOLDEN_RATIO_CONJUGATE) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
            out[name] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
    return out
