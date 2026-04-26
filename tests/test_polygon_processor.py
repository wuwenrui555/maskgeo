"""Tests for PolygonProcessor — single-polygon utilities."""
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from maskgeo.processor import PolygonProcessor


# ─── fix_geometry_to_polygon ──────────────────────────────────────────────────

def test_fix_polygon_passthrough():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert PolygonProcessor.fix_geometry_to_polygon(poly) is poly


def test_fix_multipolygon_dominant_largest():
    """MultiPolygon with one piece much larger than the other returns the largest."""
    big = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    tiny = Polygon([(200, 200), (201, 200), (201, 201), (200, 201)])
    mp = MultiPolygon([big, tiny])
    out = PolygonProcessor.fix_geometry_to_polygon(mp, multipolygon_area_ratio=100)
    # shapely 2.x re-wraps Polygons through MultiPolygon.geoms, so compare by
    # equality, not identity.
    assert out.equals(big)


def test_fix_multipolygon_balanced_returns_none():
    """MultiPolygon with comparable pieces returns None (likely user error)."""
    a = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    b = Polygon([(20, 0), (30, 0), (30, 10), (20, 10)])
    mp = MultiPolygon([a, b])
    out = PolygonProcessor.fix_geometry_to_polygon(mp, multipolygon_area_ratio=100)
    assert out is None


def test_fix_linestring_close_endpoints_closes():
    line = LineString([(0, 0), (10, 0), (10, 10), (0.1, 0.1)])  # ends near start
    out = PolygonProcessor.fix_geometry_to_polygon(line, linestring_end_distance=50)
    assert isinstance(out, Polygon)


def test_fix_linestring_far_endpoints_returns_none():
    line = LineString([(0, 0), (100, 0), (100, 100)])  # ends 100 from start
    out = PolygonProcessor.fix_geometry_to_polygon(line, linestring_end_distance=50)
    assert out is None


def test_fix_other_geometry_returns_none():
    assert PolygonProcessor.fix_geometry_to_polygon(Point(5, 5)) is None


# ─── polygon_to_mask ──────────────────────────────────────────────────────────

def test_polygon_to_mask_unit_square():
    """A polygon with corners at (1,1),(2,1),(2,2),(1,2) covers exactly pixel (1,1)."""
    poly = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    mask = PolygonProcessor.polygon_to_mask(poly, shape=(5, 5))
    expected = np.zeros((5, 5), dtype=bool)
    expected[1, 1] = True
    np.testing.assert_array_equal(mask, expected)


def test_polygon_to_mask_returns_bool():
    poly = Polygon([(0, 0), (3, 0), (3, 3), (0, 3)])
    mask = PolygonProcessor.polygon_to_mask(poly, shape=(5, 5))
    assert mask.dtype == bool


def test_polygon_to_mask_50x50_block():
    """The pixel-edge convention: corners at (30,50)..(80,100) give mask[50:100, 30:80]."""
    poly = Polygon([(30, 50), (80, 50), (80, 100), (30, 100)])
    mask = PolygonProcessor.polygon_to_mask(poly, shape=(200, 200))
    assert int(mask.sum()) == 50 * 50
    assert mask[50:100, 30:80].all()
    assert not mask[50, 29] and not mask[100, 30]  # edges respected


# ─── PolygonProcessor.__init__ ────────────────────────────────────────────────

def test_init_with_polygon_keeps_polygon():
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    pp = PolygonProcessor(poly)
    assert pp.polygon is poly


def test_init_with_multipolygon_dominant_picks_largest():
    big = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])
    tiny = Polygon([(200, 200), (201, 200), (201, 201), (200, 201)])
    pp = PolygonProcessor(MultiPolygon([big, tiny]))
    assert pp.polygon.equals(big)


def test_init_with_invalid_geometry_raises():
    with pytest.raises(ValueError, match="invalid Polygon"):
        PolygonProcessor(Point(0, 0))


# ─── crop_array_by_polygon ────────────────────────────────────────────────────

def test_crop_array_2d():
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    poly = Polygon([(2, 3), (5, 3), (5, 6), (2, 6)])
    pp = PolygonProcessor(poly)
    cropped, mask = pp.crop_array_by_polygon(img)
    assert cropped.shape == (3, 3)
    assert mask.shape == (3, 3)
    assert mask.all()  # entire crop region inside polygon
    np.testing.assert_array_equal(cropped, img[3:6, 2:5])


def test_crop_array_3d_cyx():
    img = np.zeros((4, 10, 10), dtype=np.uint8)  # 4 channels
    img[2, 3:6, 2:5] = 99  # mark channel 2 inside our polygon
    poly = Polygon([(2, 3), (5, 3), (5, 6), (2, 6)])
    pp = PolygonProcessor(poly)
    cropped, mask = pp.crop_array_by_polygon(img, dim_order="CYX")
    assert cropped.shape == (4, 3, 3)
    assert (cropped[2] == 99).all()
    assert (cropped[0] == 0).all()


def test_crop_fill_value_outside_polygon():
    """Pixels outside the polygon are replaced with fill_value."""
    img = np.full((10, 10), 200, dtype=np.uint8)
    # Triangular polygon, leaves some pixels in the bbox outside.
    poly = Polygon([(2, 3), (6, 3), (4, 6)])
    pp = PolygonProcessor(poly)
    cropped, mask = pp.crop_array_by_polygon(img, fill_value=0)
    # Where mask is False, cropped should be 0.
    assert (cropped[~mask] == 0).all()
    assert (cropped[mask] == 200).all()
