"""GeojsonProcessor + PolygonProcessor — class-based geometry handling."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from maskgeo._color import _assign_colors

logger = logging.getLogger(__name__)


# =============================================================================
# PolygonProcessor — single-polygon helpers
# =============================================================================


class PolygonProcessor:
    """Single-polygon utilities. Most users will use GeojsonProcessor instead."""

    DEFAULT_MULTIPOLYGON_AREA_RATIO: float = 100
    DEFAULT_LINESTRING_END_DISTANCE: float = 50

    def __init__(
        self,
        geometry,
        multipolygon_area_ratio: float = DEFAULT_MULTIPOLYGON_AREA_RATIO,
        linestring_end_distance: float = DEFAULT_LINESTRING_END_DISTANCE,
    ):
        polygon = self.fix_geometry_to_polygon(
            geometry, multipolygon_area_ratio, linestring_end_distance,
        )
        if polygon is None:
            raise ValueError(f"Geometry is an invalid Polygon: {geometry!r}")
        self.polygon: Polygon = polygon

    # ── geometry repair ──────────────────────────────────────────────────────

    @staticmethod
    def fix_geometry_to_polygon(
        geometry,
        multipolygon_area_ratio: float = DEFAULT_MULTIPOLYGON_AREA_RATIO,
        linestring_end_distance: float = DEFAULT_LINESTRING_END_DISTANCE,
    ) -> Optional[Polygon]:
        """Coerce a shapely geometry to a single Polygon, or return None."""
        if isinstance(geometry, Polygon):
            return geometry
        if isinstance(geometry, MultiPolygon):
            return PolygonProcessor._fix_multipolygon(geometry, multipolygon_area_ratio)
        if isinstance(geometry, LineString):
            return PolygonProcessor._fix_linestring(geometry, linestring_end_distance)
        return None

    @staticmethod
    def _fix_multipolygon(mp: MultiPolygon, area_ratio: float) -> Optional[Polygon]:
        polys = [p for p in mp.geoms if isinstance(p, Polygon)]
        if not polys:
            return None
        polys = sorted(polys, key=lambda p: p.area, reverse=True)
        if len(polys) == 1:
            return polys[0]
        ratio = polys[0].area / polys[1].area if polys[1].area else float("inf")
        if ratio >= area_ratio:
            logger.info(
                "MultiPolygon: kept largest piece (area_ratio=%.2f >= %.2f)",
                ratio, area_ratio,
            )
            return polys[0]
        return None

    @staticmethod
    def _fix_linestring(line: LineString, end_distance: float) -> Optional[Polygon]:
        start, end = Point(line.coords[0]), Point(line.coords[-1])
        d = start.distance(end)
        if d <= end_distance:
            coords = list(line.coords) + [line.coords[0]]
            logger.info(
                "LineString: closed into Polygon (end_distance=%.2f <= %.2f)",
                d, end_distance,
            )
            return Polygon(LineString(coords))
        return None

    # ── rasterization ────────────────────────────────────────────────────────

    @staticmethod
    def polygon_to_mask(polygon, shape: tuple[int, int]) -> np.ndarray:
        """Rasterize a polygon (or MultiPolygon) to a boolean mask using the pixel-edge convention."""
        mask = features.rasterize(
            [(polygon, 1)],
            out_shape=shape,
            fill=0,
            dtype=np.uint8,
            transform=Affine.identity(),
        )
        return mask.astype(bool)

    # ── cropping ─────────────────────────────────────────────────────────────

    @staticmethod
    def _crop_geometry(
        geometry,
        img: np.ndarray,
        dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop and mask `img` by an arbitrary shapely geometry (Polygon or MultiPolygon).

        Uses `rasterio.features.rasterize` with `Affine.translation(x_min, y_min)`
        so the geometry's original coordinates burn in correctly without manual
        coord shifting. Works for any geometry that has `.bounds`.

        Returns (cropped_image, mask). Pixels outside the geometry are set to
        `fill_value`.
        """
        if img.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D or 3D, got {img.ndim}D")
        if img.ndim == 3 and dim_order not in ("CYX", "YXC"):
            raise ValueError(f"dim_order must be 'CYX' or 'YXC', got {dim_order!r}")

        if img.ndim == 2:
            height, width = img.shape
        elif dim_order == "CYX":
            height, width = img.shape[1:]
        else:  # YXC
            height, width = img.shape[:2]

        x_min, y_min, x_max, y_max = geometry.bounds
        y_min = max(0, int(np.floor(y_min)))
        y_max = min(height, int(np.ceil(y_max)))
        x_min = max(0, int(np.floor(x_min)))
        x_max = min(width, int(np.ceil(x_max)))

        # Translation transform: pixel (0,0) of the cropped frame is at
        # original coord (x_min, y_min). Lets us pass the geometry as-is.
        mask = features.rasterize(
            [(geometry, 1)],
            out_shape=(y_max - y_min, x_max - x_min),
            fill=0,
            dtype=np.uint8,
            transform=Affine.translation(x_min, y_min),
        ).astype(bool)

        if img.ndim == 2:
            cropped = img[y_min:y_max, x_min:x_max].copy()
            cropped[~mask] = fill_value
            return cropped, mask

        if dim_order == "CYX":
            cropped = img[:, y_min:y_max, x_min:x_max].copy()
            cropped[:, ~mask] = fill_value
            return cropped, mask

        # YXC
        cropped = img[y_min:y_max, x_min:x_max, :].copy()
        cropped[~mask, :] = fill_value
        return cropped, mask

    def crop_array_by_polygon(
        self,
        img: np.ndarray,
        dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop and mask `img` by self.polygon. See `_crop_geometry`."""
        return PolygonProcessor._crop_geometry(self.polygon, img, dim_order, fill_value)


# =============================================================================
# GeojsonProcessor — load, sanitize, classify, crop, save
# =============================================================================


def _drop_reason(
    geom,
    multipolygon_area_ratio: float,
    linestring_end_distance: float,
) -> str:
    """Human-readable reason a geometry was dropped during sanitization."""
    if isinstance(geom, MultiPolygon):
        polys = sorted(
            [p for p in geom.geoms if isinstance(p, Polygon)],
            key=lambda p: p.area, reverse=True,
        )
        if len(polys) <= 1:
            return "MultiPolygon contained no valid sub-polygons"
        ratio = polys[0].area / polys[1].area if polys[1].area else float("inf")
        return f"MultiPolygon area_ratio {ratio:.2f} < {multipolygon_area_ratio}"
    if isinstance(geom, LineString):
        d = Point(geom.coords[0]).distance(Point(geom.coords[-1]))
        return f"LineString endpoint distance {d:.2f} > {linestring_end_distance}"
    return f"unsupported geometry type {type(geom).__name__}"


class GeojsonProcessor:
    """Load, sanitize, inspect, classify, and crop with a QuPath GeoJSON."""

    NAME_PREFIX: str = "polygon_"
    DEFAULT_CLASSIFICATION: str = json.dumps(
        {"name": "unknown", "color": [128, 128, 128]}
    )

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        polygon_only: bool = True,
        multipolygon_area_ratio: float = PolygonProcessor.DEFAULT_MULTIPOLYGON_AREA_RATIO,
        linestring_end_distance: float = PolygonProcessor.DEFAULT_LINESTRING_END_DISTANCE,
    ):
        gdf = gdf.copy().reset_index(drop=True)
        self.skipped: list[dict] = []

        # Step 1: name auto-fill (1-indexed polygon_1..polygon_N for missing).
        if "name" not in gdf.columns:
            gdf["name"] = None
        gdf["name"] = gdf["name"].astype(object)
        missing = gdf["name"].isna() | gdf["name"].astype(str).isin(["", "nan", "None"])
        if missing.any():
            n_missing = int(missing.sum())
            fill = [f"{self.NAME_PREFIX}{k+1}" for k in range(n_missing)]
            gdf.loc[missing, "name"] = fill
        gdf["name"] = gdf["name"].astype(str)

        # Default classification fallback. Normalize all entries to JSON strings
        # — gpd.read_file parses property dicts directly, while QuPath / our
        # output_geojson writes JSON strings. We store strings throughout for
        # consistency; downstream readers (plot_*, geojson_to_mask) handle both.
        if "classification" not in gdf.columns:
            gdf["classification"] = self.DEFAULT_CLASSIFICATION
        else:
            def _normalize_cls(v):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    return self.DEFAULT_CLASSIFICATION
                if isinstance(v, dict):
                    return json.dumps(v)
                return v
            gdf["classification"] = gdf["classification"].apply(_normalize_cls)

        self.gdf_raw = gdf.copy()

        # Step 2: geometry sanitize.
        kept_geoms = []
        kept_idx = []
        for i, (idx, geom) in enumerate(zip(gdf.index, gdf["geometry"])):
            if polygon_only:
                fixed = PolygonProcessor.fix_geometry_to_polygon(
                    geom, multipolygon_area_ratio, linestring_end_distance,
                )
                if fixed is None:
                    self.skipped.append({
                        "index": int(i),
                        "name": str(gdf.iloc[i]["name"]),
                        "reason": _drop_reason(
                            geom, multipolygon_area_ratio, linestring_end_distance,
                        ),
                    })
                else:
                    kept_geoms.append(fixed)
                    kept_idx.append(idx)
            else:
                # Keep Polygon + MultiPolygon as-is, close LineString, drop other.
                if isinstance(geom, (Polygon, MultiPolygon)):
                    kept_geoms.append(geom)
                    kept_idx.append(idx)
                elif isinstance(geom, LineString):
                    closed = PolygonProcessor._fix_linestring(geom, linestring_end_distance)
                    if closed is None:
                        self.skipped.append({
                            "index": int(i),
                            "name": str(gdf.iloc[i]["name"]),
                            "reason": _drop_reason(
                                geom, multipolygon_area_ratio, linestring_end_distance,
                            ),
                        })
                    else:
                        kept_geoms.append(closed)
                        kept_idx.append(idx)
                else:
                    self.skipped.append({
                        "index": int(i),
                        "name": str(gdf.iloc[i]["name"]),
                        "reason": f"unsupported geometry type {type(geom).__name__}",
                    })

        gdf = gdf.loc[kept_idx].copy()
        gdf["geometry"] = kept_geoms

        # Step 3: dedup names with cumcount suffix (only for actually-duplicated names).
        cumcount = gdf.groupby("name").cumcount() + 1
        counts = gdf["name"].value_counts()
        dups = set(counts[counts > 1].index)
        new_names = [
            f"{n}_{c}" if n in dups else n
            for n, c in zip(gdf["name"], cumcount)
        ]
        gdf.index = pd.Index(new_names, name=None)
        self.gdf = gdf

    # ── factory methods ──────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        polygon_only: bool = True,
        multipolygon_area_ratio: float = PolygonProcessor.DEFAULT_MULTIPOLYGON_AREA_RATIO,
        linestring_end_distance: float = PolygonProcessor.DEFAULT_LINESTRING_END_DISTANCE,
    ) -> "GeojsonProcessor":
        return cls(
            gpd.read_file(path),
            polygon_only=polygon_only,
            multipolygon_area_ratio=multipolygon_area_ratio,
            linestring_end_distance=linestring_end_distance,
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        polygon_only: bool = True,
        multipolygon_area_ratio: float = PolygonProcessor.DEFAULT_MULTIPOLYGON_AREA_RATIO,
        linestring_end_distance: float = PolygonProcessor.DEFAULT_LINESTRING_END_DISTANCE,
    ) -> "GeojsonProcessor":
        data = json.loads(text)
        gdf = gpd.GeoDataFrame.from_features(data["features"])
        return cls(
            gdf,
            polygon_only=polygon_only,
            multipolygon_area_ratio=multipolygon_area_ratio,
            linestring_end_distance=linestring_end_distance,
        )

    # ── classification mutation ──────────────────────────────────────────────

    def update_classification(
        self,
        name_dict: dict[str, str],
        color_dict: Optional[dict[str, list[int]]] = None,
    ) -> None:
        """Update `properties.classification` for polygons whose name is in `name_dict`.

        Auto-assigns colors via _color._assign_colors when color_dict is None.
        Polygons with names not in name_dict keep their existing classification.
        """
        cls_target_names = sorted(set(name_dict.values()))
        if color_dict is None:
            palette = _assign_colors(cls_target_names)
        else:
            palette = dict(color_dict)
            missing_cls = [n for n in cls_target_names if n not in palette]
            if missing_cls:
                palette.update(_assign_colors(missing_cls))

        unmapped = []
        for poly_name, target_cls in name_dict.items():
            if poly_name not in self.gdf.index:
                unmapped.append(poly_name)
                continue
            self.gdf.at[poly_name, "classification"] = json.dumps({
                "name": target_cls,
                "color": palette[target_cls],
            })
        if unmapped:
            logger.warning(
                "update_classification: %d name(s) in name_dict not found in "
                "gp.gdf.index (skipped): %s",
                len(unmapped), unmapped,
            )

    # ── persistence ──────────────────────────────────────────────────────────

    def output_geojson(self, path: Union[str, Path]) -> None:
        """Write self.gdf to a GeoJSON file."""
        self.gdf.to_file(path, driver="GeoJSON")

    # ── plotting ─────────────────────────────────────────────────────────────

    def plot_classification(
        self,
        figsize: tuple[float, float] = (10, 10),
        legend: bool = True,
        plot_raw: bool = False,
        ax=None,
    ):
        """Plot polygons colored by their classification color."""
        import matplotlib.pyplot as plt

        gdf = self.gdf_raw if plot_raw else self.gdf
        colors = []
        names = []
        for cls_str in gdf["classification"]:
            try:
                cls = json.loads(cls_str) if isinstance(cls_str, str) else (cls_str or {})
            except (json.JSONDecodeError, TypeError):
                cls = {}
            rgb = cls.get("color", [128, 128, 128])
            colors.append(f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}")
            names.append(cls.get("name", "unknown"))

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        gdf.plot(ax=ax, color=colors, aspect=1)

        if legend:
            seen = set()
            for n, c in zip(names, colors):
                if n in seen:
                    continue
                seen.add(n)
                ax.scatter([], [], c=c, label=n)
            ax.legend(title="Classification", loc="center left",
                      bbox_to_anchor=(1, 0.5))

        ax.invert_yaxis()
        ax.set_aspect("equal")
        return fig

    def plot_name(
        self,
        figsize: tuple[float, float] = (10, 10),
        text: bool = True,
        text_size: int = 12,
        text_color: str = "black",
        ax=None,
    ):
        """Plot polygons with their name shown at each polygon's centroid."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        unique_names = list(dict.fromkeys(self.gdf.index))
        palette = _assign_colors(unique_names)
        colors = [
            f"#{palette[n][0]:02x}{palette[n][1]:02x}{palette[n][2]:02x}"
            for n in self.gdf.index
        ]
        self.gdf.plot(ax=ax, color=colors, aspect=1)

        if text:
            for name, geom in zip(self.gdf.index, self.gdf.geometry):
                x_min, y_min, x_max, y_max = geom.bounds
                ax.text(
                    (x_min + x_max) / 2, (y_min + y_max) / 2, name,
                    fontsize=text_size, ha="center", va="center", color=text_color,
                )

        ax.invert_yaxis()
        ax.set_aspect("equal")
        return fig

    # ── cropping ─────────────────────────────────────────────────────────────

    def crop_image(
        self,
        img,
        dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> Generator[tuple[str, object], None, None]:
        """Yield (polygon_name, cropped) for each polygon in self.gdf.

        Output type follows input type:
        - 2D ndarray  → cropped 2D ndarray
        - 3D ndarray  → cropped 3D ndarray (dim_order respected)
        - dict[str, 2D ndarray] → dict[str, 2D ndarray] for each polygon
        """
        if isinstance(img, dict):
            shapes = {v.shape for v in img.values()}
            if len({s for s in shapes if len(s) == 2}) != 1:
                raise ValueError("All arrays in dict must be 2D and same shape")
            for name, geom in zip(self.gdf.index, self.gdf.geometry):
                yield name, {
                    chan: PolygonProcessor._crop_geometry(
                        geom, arr, fill_value=fill_value,
                    )[0]
                    for chan, arr in img.items()
                }
            return

        if not isinstance(img, np.ndarray):
            raise TypeError(f"img must be np.ndarray or dict, got {type(img).__name__}")

        for name, geom in zip(self.gdf.index, self.gdf.geometry):
            cropped, _ = PolygonProcessor._crop_geometry(
                geom, img, dim_order=dim_order, fill_value=fill_value,
            )
            yield name, cropped
