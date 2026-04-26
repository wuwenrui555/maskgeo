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

# "Unknown" / "unknown" is reserved for unclassified regions and always maps to
# neutral gray, regardless of where it falls in the input list.
UNKNOWN_COLOR: list[int] = [128, 128, 128]
_UNKNOWN_NAMES: frozenset[str] = frozenset({"Unknown", "unknown"})

# Golden-ratio conjugate for hue stepping when n > 6.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _assign_colors(names: list[str]) -> dict[str, list[int]]:
    """Deterministically assign distinct RGB colors to a list of names.

    "Unknown" / "unknown" always map to neutral gray ([128, 128, 128]) and do
    not consume a slot in the primary palette. The remaining names get the 6
    primaries first (red/green/blue/yellow/magenta/cyan), then HSV hue stepping
    by the golden ratio conjugate.

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
    palette_index = 0
    for name in names:
        if name in _UNKNOWN_NAMES:
            out[name] = list(UNKNOWN_COLOR)
            continue
        if palette_index < len(_PRIMARY_COLORS):
            out[name] = list(_PRIMARY_COLORS[palette_index])
        else:
            hue = (
                (palette_index - len(_PRIMARY_COLORS) + 1) * _GOLDEN_RATIO_CONJUGATE
            ) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
            out[name] = [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]
        palette_index += 1
    return out
