# maskgeo — design spec

**Date:** 2026-04-26
**Status:** Approved (brainstorming complete, ready for implementation plan)
**Author:** Wenrui Wu

## 1. Summary

`maskgeo` is a focused Python library that converts between segmentation masks
(numpy arrays) and pixel-edge polygons (GeoJSON). It supersedes the
`geojson.py` portion of the legacy `pyqupath` package, builds on `rasterio.features`
for the core conversion, and depends on `pyratiff` for OME-TIFF I/O.

**Design north star:** one focused job, done correctly, with API that matches
how QuPath / ImageJ / GIS tools actually represent rasters as polygons (the
pixel-as-area / pixel-edge convention).

## 2. Motivation

The existing `pyqupath` package (v0.2.0) has three structural problems:

1. **Reinvents what `rasterio.features` already does.** The 600+ line
   `binary_mask_to_polygon` family pads masks with `+1`, runs `cv2.findContours`
   on the padded grid, then walks any diagonal contour segments to insert
   axis-aligned intermediates. Empirical testing in QuPath (see `tests/data/`
   fixtures captured during the brainstorming session) confirms that
   `rasterio.features.shapes(mask, transform=Affine.identity())` produces the
   same QuPath-correct output natively, in ~5 lines of glue.

2. **Duplicate TIFF implementations.** `pyqupath.tiff` and `pyqupath.ometiff`
   are two parallel implementations of pyramidal OME-TIFF I/O. The clean OO
   version was already extracted into the standalone `pyratiff` package
   (v1.0.0). `maskgeo` depends on it rather than copying.

3. **Pinned dependencies block modern Python.** `pandas<2`, `numpy<2`,
   `tifffile<2024.9.20`, `zarr<3` mean `pip install pyqupath` fails on
   Python 3.12+ because pandas 1.5.3 cannot build (missing `pkg_resources`).
   `maskgeo` mirrors `pyratiff`'s `>=` floors only.

## 3. Scope

### In scope (v1.0.0)

- `mask_to_geojson(mask, path, ...)` — labeled mask → QuPath-compatible GeoJSON
- `geojson_to_mask(path, shape, ...)` — GeoJSON → labeled mask
- `GeojsonProcessor` — load / sanitize / inspect / classify / crop / save GeoJSON
- `PolygonProcessor` — single-polygon helpers (rasterize, crop, geometry fix)
- Internal `_color.py` for automatic classification colors

### Out of scope (intentionally cut)

- TMA dearrayer (`pyqupath/tma.py`, 786 lines) — independent functionality;
  candidate for a future standalone `pyrama` package
- Open-polyline merging (`pyqupath/buffer.py`, 244 lines) — no current callers,
  YAGNI
- All `pyqupath/ometiff.py` content — replaced by `pyratiff.PyramidWriter`
- All `pyqupath/tiff.py` content — replaced by `pyratiff.TiffZarrReader`
- Joblib-based parallel `mask_to_polygons` family —
  `rasterio.features.shapes` runs in C and is fast enough for cell-level
  segmentations
- All deprecated functions (`load_geojson_to_gdf`, `update_geojson_classification`,
  `crop_dict_by_geojson`, `binary_mask_to_polygon`, `mask_to_polygon_batch`,
  `mask_to_polygons`, `mask_to_geojson_joblib`)

### Explicitly NOT a goal

- Migrating existing pyqupath callers (itercluster, jiale_review, inframe scripts).
  Those keep working with `pyqupath==0.2.0` from PyPI. Migration is voluntary
  and lazy.

## 4. Pixel-edge convention (load-bearing)

`maskgeo` standardizes on the **pixel-edge / pixel-as-area** convention
(QuPath, ImageJ, GIS, GeoJSON, rasterio). A pixel at array index `(r, c)`
covers the area `[c, c+1) × [r, r+1)` in geometric coordinates. Integer
coordinates represent pixel boundaries, not pixel centers.

This is opposite to OpenCV's `cv2.findContours` and scikit-image's
`find_contours`, which return contours where integer coordinates represent
pixel centers.

**Empirical validation** (see `tests/data/rect.{ome.tiff,geojson}` and
`tests/test_pixel_edge.py`): a 50x50 mask block at `mask[50:100, 30:80]`
round-trips through `rasterio.features.shapes` → `rasterio.features.rasterize`
with **0 pixel difference**, and the resulting polygon
`[(30, 50), (30, 100), (80, 100), (80, 50), (30, 50)]` aligns flush with the
mask's white region in QuPath.

This convention is automatic when both calls use `transform=Affine.identity()`,
which is the default `maskgeo` always uses.

## 5. Public API

`maskgeo/__init__.py` exports exactly four public symbols:

```python
from maskgeo.convert import mask_to_geojson, geojson_to_mask
from maskgeo.processor import GeojsonProcessor, PolygonProcessor
```

### 5.1 `mask_to_geojson`

```python
def mask_to_geojson(
    mask: np.ndarray,
    geojson_path: str | Path,
    annotation_dict: dict[int, str] | None = None,
    color_dict: dict[str, list[int]] | None = None,
    simplify_tolerance: float | None = None,
) -> None:
    """Convert a labeled mask to a QuPath-compatible GeoJSON file.

    Each unique non-zero label in the mask becomes one Feature. If a label has
    multiple disjoint connected components, the Feature uses MultiPolygon
    geometry.

    Parameters
    ----------
    mask : np.ndarray
        2D integer array. 0 is background; positive integers are labels.
    geojson_path : str or Path
        Output file path.
    annotation_dict : dict[int, str], optional
        Maps label values to classification names (e.g., {1: "Tumor"}).
        Labels not in the dict are classified as "Unknown".
    color_dict : dict[str, list[int]], optional
        Maps classification names to RGB colors as 3-element lists.
        Names not in the dict get auto-assigned colors via _assign_colors.
    simplify_tolerance : float, optional
        If set, simplify polygons via shapely's .simplify(tolerance) before
        writing. Smaller values retain more detail.
    """
```

**Output Feature schema:**

```json
{
  "type": "Feature",
  "geometry": { "type": "Polygon" | "MultiPolygon", "coordinates": [...] },
  "properties": {
    "objectType": "annotation",
    "name": "1",
    "classification": { "name": "Tumor", "color": [255, 0, 0] }
  }
}
```

`properties.name` is always `str(label)`. This enables round-trip with
`geojson_to_mask(label_by="name")`.

### 5.2 `geojson_to_mask`

```python
def geojson_to_mask(
    geojson_path: str | Path,
    shape: tuple[int, int],
    label_dict: dict[str, int] | None = None,
    label_by: str = "name",
    polygon_only: bool = True,
    multipolygon_area_ratio: float = 100,
    linestring_end_distance: float = 50,
) -> np.ndarray:
    """Rasterize a GeoJSON file into a labeled mask.

    Parameters
    ----------
    geojson_path : str or Path
    shape : (H, W) of the output mask, in pixel coordinates.
    label_dict : dict[str, int], optional
        Explicit mapping from name to integer label. Polygons whose name is
        not in the dict are skipped.
    label_by : "name" or "classification"
        - "name" (default): label per polygon comes from polygon.name
          (i.e. gp.gdf.index after sanitization).
          - If label_dict provided: label = label_dict[name]
          - Else: int(name) for each name if all parse as int; otherwise
            positional 1..N
        - "classification": label per polygon comes from
          properties.classification.name. label_dict is required.
    polygon_only : bool, default True
        Forwarded to GeojsonProcessor.from_path. Default True (strict) means
        MultiPolygon is reduced to its largest piece if the largest dominates,
        otherwise dropped. Set False to keep MultiPolygon as-is (needed for
        round-trip from mask_to_geojson when labels span disjoint regions).
    multipolygon_area_ratio : float, default 100
        Used only when polygon_only=True.
    linestring_end_distance : float, default 50
        Used in both modes.

    Returns
    -------
    np.ndarray
        Mask of shape `shape`, dtype uint8 if max label < 256 else int32,
        with 0 for background and positive integers for polygon labels.
    """
```

**Round-trip requirement:** `mask_to_geojson` may produce MultiPolygon
features (one label, disjoint pieces). To round-trip, callers must pass
`polygon_only=False`:

```python
mask_to_geojson(cell_mask, "cells.geojson")
back = geojson_to_mask("cells.geojson", shape=cell_mask.shape, polygon_only=False)
assert (back == cell_mask).all()
```

### 5.3 `GeojsonProcessor`

```python
class GeojsonProcessor:
    """Load, sanitize, inspect, classify, and crop with a QuPath GeoJSON."""

    NAME_PREFIX: ClassVar[str] = "polygon_"

    def __init__(
        self,
        gdf: gpd.GeoDataFrame,
        polygon_only: bool = True,
        multipolygon_area_ratio: float = 100,
        linestring_end_distance: float = 50,
    ): ...

    @classmethod
    def from_path(cls, path, polygon_only=True, ...) -> "GeojsonProcessor": ...

    @classmethod
    def from_text(cls, text, polygon_only=True, ...) -> "GeojsonProcessor": ...

    # Data
    gdf: gpd.GeoDataFrame          # sanitized, indexed by unique name
    gdf_raw: gpd.GeoDataFrame      # original, before sanitization
    skipped: list[dict]            # per-feature reason for being dropped

    # Classification
    def update_classification(
        self,
        name_dict: dict[str, str],
        color_dict: dict[str, list[int]] | None = None,
    ) -> None: ...

    # Plotting (matplotlib)
    def plot_classification(
        self, figsize=(10, 10), legend=True, plot_raw=False, ax=None,
    ) -> "plt.Figure": ...

    def plot_name(
        self, figsize=(10, 10), text=True, text_size=12,
        text_color="black", ax=None,
    ) -> "plt.Figure": ...

    # Cropping
    def crop_image(
        self,
        img: np.ndarray | dict[str, np.ndarray],
        dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> Generator[tuple[str, ...], None, None]:
        """Yield (polygon_name, cropped) for each polygon in gp.gdf.

        Output type follows input:
        - 2D ndarray  → cropped 2D ndarray
        - 3D ndarray  → cropped 3D ndarray (dim_order respected)
        - dict        → dict[str, 2D ndarray] for each polygon
        """

    # Output
    def output_geojson(self, path: str | Path) -> None: ...
```

#### 5.3.1 Name normalization (in `__init__`)

Three steps, in this order:

1. **Auto-fill missing names.** Features with no `properties.name` (or NaN /
   empty) get `f"{NAME_PREFIX}{k}"` where `k` is the 1-indexed position of
   the missing-name feature among missing-name features (so the assigned
   names are `polygon_1`, `polygon_2`, ..., `polygon_N`, never `polygon_0`).
2. **Geometry sanitize** (see §5.3.2).
3. **Deduplicate.** Names appearing more than once get cumcount suffixes:
   `"Tumor"`, `"Tumor"` → `"Tumor_1"`, `"Tumor_2"`. The result becomes
   `gp.gdf.index`.

After init, `gp.gdf.index` is guaranteed to be a list of unique strings.

#### 5.3.2 Geometry sanitization rules

| Input geometry      | `polygon_only=True` (default)               | `polygon_only=False`        |
|---------------------|---------------------------------------------|-----------------------------|
| Polygon             | keep                                        | keep                        |
| MultiPolygon        | largest piece if `largest/second >= ratio`; otherwise drop | keep |
| LineString (close)  | close into Polygon                          | close into Polygon          |
| LineString (far)    | drop                                        | drop                        |
| other               | drop                                        | drop                        |

Every **dropped** feature (those that did not survive sanitization) is
appended to `gp.skipped` with shape:

```python
{"index": int, "name": str, "reason": str}
```

Modifications that keep the feature (LineString closed, MultiPolygon reduced
to largest dominant piece) are reported via Python `logging` at `INFO`
level, not added to `gp.skipped`. This replaces pyqupath's `print()` noise
and lets callers inspect drops programmatically while keeping the
"successfully kept" path quiet by default.

#### 5.3.3 `update_classification`

Updates `properties.classification.{name, color}` in `gp.gdf` based on the
current `name` (i.e. polygon name). When `color_dict` is None, colors are
auto-assigned via `_assign_colors` deterministically by classification name.
Polygons whose name is not in `name_dict` keep their existing classification.

### 5.4 `PolygonProcessor`

```python
class PolygonProcessor:
    """Single-polygon utilities. Most users will use GeojsonProcessor instead."""

    def __init__(self, geometry: BaseGeometry,
                 multipolygon_area_ratio: float = 100,
                 linestring_end_distance: float = 50): ...

    polygon: Polygon  # always a valid Polygon after init

    @staticmethod
    def fix_geometry_to_polygon(
        geometry: BaseGeometry,
        multipolygon_area_ratio: float = 100,
        linestring_end_distance: float = 50,
    ) -> Polygon | None: ...

    @staticmethod
    def polygon_to_mask(
        polygon: Polygon, shape: tuple[int, int],
    ) -> np.ndarray:
        """Rasterize a single polygon to a boolean mask via rasterio."""

    def crop_array_by_polygon(
        self, img: np.ndarray, dim_order: str = "CYX",
        fill_value: float = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (masked_image, mask)."""
```

## 6. Module layout

```
maskgeo/
├── src/maskgeo/
│   ├── __init__.py        # explicit public API exports
│   ├── _color.py          # internal: _assign_colors
│   ├── convert.py         # mask_to_geojson, geojson_to_mask
│   └── processor.py       # GeojsonProcessor, PolygonProcessor
├── tests/
│   ├── conftest.py
│   ├── data/
│   │   ├── rect.ome.tiff
│   │   ├── rect.geojson
│   │   ├── multilabel.geojson
│   │   └── multilabel_qupath_export.geojson
│   ├── test_convert.py
│   ├── test_processor.py
│   └── test_pixel_edge.py
├── docs/
│   └── superpowers/specs/    # design + plan docs
├── pyproject.toml
├── README.md
├── CHANGELOG.md
├── LICENSE
└── .gitignore
```

## 7. Dependencies and project configuration

### `pyproject.toml`

```toml
[project]
name = "maskgeo"
version = "1.0.0"
description = "Convert between segmentation masks and pixel-edge polygons (GeoJSON)"
readme = "README.md"
license = "GPL-3.0-or-later"
authors = [{ name = "wuwenrui555", email = "wuwenruiwwr@outlook.com" }]
requires-python = ">=3.12"
dependencies = [
    "numpy>=1.21",
    "rasterio>=1.4",
    "shapely>=2.0",
    "geopandas>=1.0",
    "pyratiff>=1.0",
    "matplotlib>=3",
]

[dependency-groups]
dev = [
    "pytest>=8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/maskgeo"]
```

**Dropped from pyqupath:** `opencv-python`, `joblib`, `tqdm-joblib`, `tqdm`,
`imagecodecs`, `tifffile<2024.9.20`, `zarr<3`, `scikit-image<0.20`,
`pandas<2`. All upper bounds removed; only `>=` floors remain.

**Why `matplotlib` is a hard dependency, not optional:**
`GeojsonProcessor.plot_classification` and `plot_name` are public API used
during interactive QC. Lazy-import would add a usage gotcha. Revisit only
if a real caller complains about import weight.

## 8. Testing strategy

`pytest`-based, all tests must pass before v1.0.0 release.

### `test_convert.py`

- `test_mask_to_geojson_roundtrip` — generated multi-label mask → geojson →
  mask, assert 0 pixel difference (uses `polygon_only=False`)
- `test_mask_to_geojson_multipolygon` — label 3 with 2 disjoint blocks
  produces a single Feature with MultiPolygon geometry
- `test_mask_to_geojson_simplify` — `simplify_tolerance` reduces vertex count
- `test_geojson_to_mask_default_name_intlike` — `label_by="name"` parses int
  names back to original labels
- `test_geojson_to_mask_default_name_positional` — `label_by="name"` falls
  back to positional when names are non-int strings (e.g., "polygon_1")
- `test_geojson_to_mask_classification` — `label_by="classification"` with
  explicit `label_dict` groups polygons by classification.name
- `test_geojson_to_mask_label_dict_subset` — polygons whose name is not in
  `label_dict` are skipped
- `test_geojson_to_mask_qupath_export` — uses real QuPath-exported geojson
  with no `name` and no `classification`; should auto-fill names and assign
  positional labels

### `test_processor.py`

- `test_from_path_auto_fill_names` — features missing `name` get
  `polygon_1..N`
- `test_from_path_dedup_names` — duplicate names get `_1`, `_2` suffixes
- `test_from_path_polygon_only_drops_balanced_multipolygon` —
  `polygon_only=True` with `area_ratio=100` drops a MultiPolygon whose two
  parts have similar areas
- `test_from_path_polygon_only_keeps_dominant_multipolygon` — same but with
  area ratio 1000 → largest piece kept
- `test_from_path_polygon_only_false_keeps_multipolygon` —
  `polygon_only=False` keeps MultiPolygon untouched
- `test_from_path_closes_near_linestring` — open polyline with close
  endpoints becomes Polygon
- `test_from_path_drops_far_linestring` — open polyline with far endpoints
  goes to `skipped`
- `test_skipped_attribute_format` — `gp.skipped` is a list of dicts with
  `index`, `name`, `reason`
- `test_update_classification_auto_color` — names get distinct colors via
  `_assign_colors`
- `test_update_classification_explicit_color` — provided `color_dict`
  overrides auto-assignment
- `test_crop_image_2d` — 2D ndarray input → 2D ndarray output
- `test_crop_image_3d_cyx` — 3D ndarray input with `dim_order="CYX"`
- `test_crop_image_dict` — dict input → dict output
- `test_output_geojson_round_trip` — write then re-load equals original gdf

### `test_pixel_edge.py`

- `test_rasterio_identity_aligns_with_qupath` — rebuild the
  `tests/data/rect.{ome.tiff,geojson}` fixture and verify the geojson
  polygon coordinates exactly match `[(30, 50), (30, 100), (80, 100),
  (80, 50)]` for a `mask[50:100, 30:80] = 1` block
- `test_single_polygon_roundtrip` — random polygon → mask via
  `polygon_to_mask` → polygon via `mask_to_geojson` → mask, assert 0 diff

### Test fixtures (in `tests/data/`)

- `rect.ome.tiff` + `rect.geojson` — single 50x50 white rectangle (the
  Section 4 validation case)
- `multilabel.ome.tiff` + `multilabel.geojson` — 4 labels including a
  MultiPolygon and a staircase
- `multilabel_qupath_export.geojson` — real QuPath export with no `name`
  and no `classification` (rectangles + ellipses with `isEllipse: true`
  flag)

## 9. Migration plan for existing pyqupath callers

**No code changes for existing callers as part of this work.** They keep
using `pyqupath==0.2.0` from PyPI.

For reference, the migration mapping (to be documented in `pyqupath/MIGRATION.md`):

| Caller (untouched) | Old import | New import (when caller chooses to migrate) |
|---|---|---|
| `inframe_article/script/alignment_performance/00_data_preparation.py` | `from pyqupath.tiff import PyramidWriter, TiffZarrReader` | `from pyratiff import PyramidWriter, TiffZarrReader` |
| `inframe_article/script/alignment_performance/04_reconstuct.py` | `from pyqupath.tiff import PyramidWriter` | `from pyratiff import PyramidWriter` |
| `itercluster/integration/qupath.py` | `from pyqupath.geojson import GeojsonProcessor` | `from maskgeo import GeojsonProcessor` |
| `jiale_review/CRCASCOLT_tumorannotations_fromgeojson.py` | `from pyqupath.geojson import GeojsonProcessor, PolygonProcessor` | `from maskgeo import GeojsonProcessor`; replace `PolygonProcessor.polygon_to_mask` calls with `geojson_to_mask(path, shape).astype(bool)` |

### Old pyqupath end-of-life

- Add `MIGRATION.md` to the pyqupath repo (root level)
- Add a deprecation banner to `pyqupath/README.md`
- Do not delete `pyqupath==0.2.0` from PyPI
- Do not publish a new pyqupath release

## 10. Implementation order

Each step is one TDD cycle (test red → implement → test green) using the
`executing-plans-test-first` workflow.

1. Repo scaffold: `pyproject.toml`, `src/maskgeo/__init__.py` empty,
   `tests/conftest.py`, CI-ready
2. `convert.py`: `mask_to_geojson` + `geojson_to_mask` + full round-trip and
   pixel-edge tests
3. `_color.py`: `_assign_colors` (deterministic, primary-colors-first)
4. `processor.py` `GeojsonProcessor.__init__` / `from_path` / `from_text`:
   name normalization + geometry sanitize + `polygon_only` + `gp.skipped`
5. `processor.py` `GeojsonProcessor` continued: `update_classification`,
   `output_geojson`, `plot_classification`, `plot_name`
6. `processor.py` `GeojsonProcessor.crop_image`: dispatch on input type
7. `processor.py` `PolygonProcessor`: `fix_geometry_to_polygon`,
   `polygon_to_mask`, `crop_array_by_polygon`
8. `README.md` + `CHANGELOG.md` for v1.0.0
9. pyqupath repo: `MIGRATION.md` + deprecation banner (separate commit
   in the pyqupath repo, not maskgeo)

## 11. Release

- Tag `v1.0.0` on `main`
- `uv build && uv publish` to PyPI
- GitHub release with CHANGELOG content

No alpha / beta / rc — the API is already validated via the prototype tests
in `jiale_review/test_design.py`.

## 12. Long-term maintenance

- **Do not add features** unless a real caller requests one
- TMA dearrayer, polyline merging, and similar pyqupath features go to
  separate packages if they ever materialize a need
- Track upstream dep upgrades (`pyratiff` future `zarr 4` /
  `tifffile` releases): bump `>=` floors only when a caller hits a wall

## 13. Open questions

None. All decisions resolved during 2026-04-26 brainstorming session
(captured in conversation log; key empirical validations preserved as
test fixtures in `tests/data/`).
