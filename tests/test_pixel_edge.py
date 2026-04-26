"""Empirical tests proving rasterio.features alignment with QuPath's pixel-edge convention."""
import json

import numpy as np

from maskgeo.convert import mask_to_geojson
from maskgeo.processor import PolygonProcessor


def test_50x50_block_polygon_corners_align_to_pixel_edges(tmp_path):
    """A 50x50 mask block at rows[50:100], cols[30:80] produces a polygon with
    corners at exactly (30,50)-(30,100)-(80,100)-(80,50)."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[50:100, 30:80] = 1
    out = tmp_path / "rect.geojson"
    mask_to_geojson(mask, out)
    data = json.loads(out.read_text())
    coords = data["features"][0]["geometry"]["coordinates"][0]
    corners_set = {tuple(c) for c in coords}
    assert corners_set == {(30, 50), (30, 100), (80, 100), (80, 50)}


def test_single_polygon_round_trip_preserves_pixels():
    """polygon_to_mask → mask → polygon_to_mask gives back the exact same mask."""
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import Polygon

    poly = Polygon([(30, 50), (80, 50), (80, 100), (30, 100)])
    mask = PolygonProcessor.polygon_to_mask(poly, shape=(200, 200))
    # Reconstruct the polygon from the mask via rasterio.features.shapes
    # then re-rasterize and compare.
    out_geoms = list(features.shapes(mask.astype(np.uint8),
                                     transform=Affine.identity()))
    fg = [shp for shp, val in out_geoms if val == 1]
    assert len(fg) == 1
    re_mask = features.rasterize(
        [(fg[0], 1)], out_shape=mask.shape, fill=0,
        dtype=np.uint8, transform=Affine.identity(),
    ).astype(bool)
    np.testing.assert_array_equal(re_mask, mask)
