"""Function-style mask <-> GeoJSON conversion built on rasterio.features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, mapping, shape

from maskgeo._color import _assign_colors
from maskgeo.processor import GeojsonProcessor, PolygonProcessor


def mask_to_geojson(
    mask: np.ndarray,
    geojson_path: Union[str, Path],
    annotation_dict: Optional[dict[int, str]] = None,
    color_dict: Optional[dict[str, list[int]]] = None,
    simplify_tolerance: Optional[float] = None,
) -> None:
    """Convert a labeled mask to a QuPath-compatible GeoJSON file.

    Each unique non-zero label becomes one Feature. Disjoint pieces of the
    same label produce a MultiPolygon geometry. Coordinates use the pixel-edge
    convention via rasterio.features.shapes with identity transform.
    """
    annotation_dict = annotation_dict or {}

    # Validate input shape and dtype.
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got {mask.ndim}D shape={mask.shape}")
    if mask.dtype.kind == "b":
        mask = mask.astype(np.uint8)
    elif mask.dtype.kind not in ("i", "u"):
        raise TypeError(
            f"mask must have integer dtype (got {mask.dtype}). "
            "Floating-point masks are ambiguous; cast explicitly first."
        )

    # "Unknown" is special-cased to gray inside _assign_colors; all other names
    # fill the primary palette first, then golden-ratio hue stepping.
    cls_names = sorted(set(annotation_dict.values()) | {"Unknown"})
    auto = _assign_colors(cls_names)
    colors = {**auto, **(color_dict or {})}

    # Group polygons by label so disjoint pieces become MultiPolygon.
    label_to_geoms: dict[int, list] = {}
    for geom_dict, value in features.shapes(
        mask.astype(np.int32), transform=Affine.identity()
    ):
        label = int(value)
        if label == 0:
            continue
        geom = shape(geom_dict)
        if simplify_tolerance is not None:
            geom = geom.simplify(simplify_tolerance)
        if geom.is_valid and not geom.is_empty:
            label_to_geoms.setdefault(label, []).append(geom)

    feats = []
    for label in sorted(label_to_geoms):
        geoms = label_to_geoms[label]
        geom = geoms[0] if len(geoms) == 1 else MultiPolygon(geoms)
        cls_name = annotation_dict.get(label, "Unknown")
        feats.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {
                "objectType": "annotation",
                "name": str(label),
                "classification": {"name": cls_name, "color": colors[cls_name]},
            },
        })

    Path(geojson_path).write_text(
        json.dumps({"type": "FeatureCollection", "features": feats})
    )


def geojson_to_mask(
    geojson_path: Union[str, Path],
    shape: tuple[int, int],
    label_dict: Optional[dict[str, int]] = None,
    label_by: str = "name",
    polygon_only: bool = True,
    multipolygon_area_ratio: float = PolygonProcessor.DEFAULT_MULTIPOLYGON_AREA_RATIO,
    linestring_end_distance: float = PolygonProcessor.DEFAULT_LINESTRING_END_DISTANCE,
) -> np.ndarray:
    """Rasterize a GeoJSON file to a labeled mask.

    Parameters
    ----------
    geojson_path : str or Path
    shape : (H, W) of the output mask in pixel coordinates.
    label_dict : dict[str, int], optional
        - label_by="name": map polygon name → integer label.
        - label_by="classification": map classification name → integer label
          (required in this mode).
    label_by : "name" | "classification"
        - "name" (default): label per polygon comes from polygon name.
          If label_dict is None, parse all names as int when possible;
          otherwise positional 1..N.
        - "classification": label per polygon comes from
          properties.classification.name. label_dict required.
    polygon_only : bool, default True
        Pass False to keep MultiPolygon as-is (round-trip case).
    """
    if label_by not in ("name", "classification"):
        raise ValueError(
            f"label_by must be 'name' or 'classification', got {label_by!r}"
        )
    if label_by == "classification" and label_dict is None:
        raise ValueError("label_by='classification' requires an explicit label_dict")

    gp = GeojsonProcessor.from_path(
        geojson_path,
        polygon_only=polygon_only,
        multipolygon_area_ratio=multipolygon_area_ratio,
        linestring_end_distance=linestring_end_distance,
    )

    H, W = shape
    if len(gp.gdf) == 0:
        return np.zeros((H, W), dtype=np.uint8)

    if label_by == "name":
        names = list(gp.gdf.index)
        if label_dict is not None:
            labels = [label_dict.get(n) for n in names]
        else:
            all_int = all(n.isdigit() for n in names)
            labels = (
                [int(n) for n in names] if all_int
                else list(range(1, len(names) + 1))
            )
    else:  # classification
        cls_names: list = []
        for cls_str in gp.gdf["classification"]:
            if isinstance(cls_str, dict):
                cls_names.append(cls_str.get("name"))
            elif isinstance(cls_str, str):
                try:
                    cls_names.append(json.loads(cls_str).get("name"))
                except json.JSONDecodeError:
                    cls_names.append(None)
            else:
                cls_names.append(None)
        labels = [label_dict.get(c) for c in cls_names]

    pairs = [
        (geom, lab)
        for geom, lab in zip(gp.gdf.geometry, labels)
        if lab is not None
    ]
    if not pairs:
        return np.zeros((H, W), dtype=np.uint8)

    max_label = max(lab for _, lab in pairs)
    dtype = np.uint8 if max_label < 256 else np.int32
    return features.rasterize(
        pairs, out_shape=(H, W), fill=0,
        transform=Affine.identity(), dtype=dtype,
    )
