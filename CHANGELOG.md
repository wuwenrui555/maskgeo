# Changelog

## [1.0.0] - 2026-04-26

Initial release. Replaces the `geojson` portion of the legacy
[`pyqupath`](https://github.com/wuwenrui555/pyqupath) package; `pyqupath` is
archived going forward.

### Added

- `mask_to_geojson(mask, path, annotation_dict, color_dict, simplify_tolerance)`
  — labeled mask → QuPath-compatible GeoJSON via `rasterio.features.shapes`
- `geojson_to_mask(path, shape, label_dict, label_by, polygon_only,
  multipolygon_area_ratio, linestring_end_distance)` — GeoJSON → labeled mask
  via `rasterio.features.rasterize`. Supports `label_by="name"` (default) and
  `label_by="classification"`.
- `GeojsonProcessor` — load (`from_path`, `from_text`), sanitize geometries,
  auto-fill missing names with `polygon_1..polygon_N`, deduplicate names with
  cumcount suffixes, inspect drops via `gp.skipped`, update classification,
  plot, crop, save.
- `PolygonProcessor` — single-polygon utilities: `fix_geometry_to_polygon`,
  `polygon_to_mask`, `crop_array_by_polygon`.

### Conventions

- Pixel-edge / pixel-as-area throughout, validated against QuPath.

### Dependencies

- numpy >= 1.21
- rasterio >= 1.4
- shapely >= 2.0
- geopandas >= 1.0
- pyratiff >= 1.0 (installed from git)
- matplotlib >= 3
