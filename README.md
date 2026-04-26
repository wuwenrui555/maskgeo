# maskgeo

Convert between segmentation masks (numpy arrays) and pixel-edge polygons
(GeoJSON). Compatible with **QuPath**, **ImageJ**, **rasterio**, and any
tool that uses the pixel-as-area convention.

## Install

```bash
pip install git+https://github.com/wuwenrui555/maskgeo.git
```

Or with uv:

```bash
uv add git+https://github.com/wuwenrui555/maskgeo.git
```

> **Note:** `rasterio` requires the GDAL system library. On Linux:
> `sudo apt install gdal-bin libgdal-dev`. On macOS: `brew install gdal`.
> Wheels for common platforms ship with GDAL bundled, so most users won't
> need to install it manually.

## Quick start

### mask → GeoJSON

```python
import numpy as np
from maskgeo import mask_to_geojson

mask = np.zeros((200, 200), dtype=np.int32)
mask[10:50, 20:60] = 1
mask[80:120, 100:140] = 2

mask_to_geojson(
    mask,
    "annotations.geojson",
    annotation_dict={1: "Tumor", 2: "Stroma"},
)
```

The resulting `annotations.geojson` opens directly in QuPath via
`File → Object data → Import objects`.

### GeoJSON → mask

```python
from maskgeo import geojson_to_mask

# Default: each polygon gets its own integer label.
mask = geojson_to_mask("annotations.geojson", shape=(200, 200))

# Group by classification name.
mask = geojson_to_mask(
    "annotations.geojson",
    shape=(200, 200),
    label_by="classification",
    label_dict={"Tumor": 1, "Stroma": 2},
)
```

### Round-trip

```python
mask_to_geojson(cell_mask, "cells.geojson")

# polygon_only=False is required to preserve MultiPolygon labels (one label
# spanning multiple disjoint regions).
back = geojson_to_mask("cells.geojson", shape=cell_mask.shape, polygon_only=False)
assert (back == cell_mask).all()
```

### Geometry sanitization

```python
from maskgeo import GeojsonProcessor

gp = GeojsonProcessor.from_path("manual.geojson")
print(f"Loaded {len(gp.gdf)}, skipped {len(gp.skipped)}")
for s in gp.skipped:
    print(f"  - {s['name']}: {s['reason']}")
```

`polygon_only=True` (default) drops MultiPolygon features whose pieces have
comparable areas (likely an accidental self-intersection during manual
drawing) and closes near-circular open polylines.

## Pixel-edge convention

`maskgeo` standardizes on the pixel-edge convention used by QuPath, ImageJ,
and the GIS world: a pixel at array index `(r, c)` covers the area
`[c, c+1) × [r, r+1)`, with integer coordinates representing pixel
boundaries. This is opposite to OpenCV's `cv2.findContours`, which uses
pixel-center coordinates.

This convention is automatic via `rasterio.features.shapes` and
`rasterio.features.rasterize` with `transform=Affine.identity()`, which
`maskgeo` always uses internally.

## Companion package

For pyramidal OME-TIFF I/O, see [`pyratiff`](https://github.com/wuwenrui555/pyratiff).

```python
from pyratiff import TiffZarrReader
from maskgeo import GeojsonProcessor

reader = TiffZarrReader.from_ometiff("image.ome.tiff")
gp = GeojsonProcessor.from_path("annotations.geojson")
for name, cropped in gp.crop_image(reader.zimg, dim_order="CYX"):
    ...
```

## License

Apache-2.0
