"""Tests for mask_to_geojson and geojson_to_mask."""
import json

import numpy as np
import pytest

from maskgeo.convert import geojson_to_mask, mask_to_geojson


# ─── output shape (mask_to_geojson) ──────────────────────────────────────────

def test_writes_featurecollection(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    out = tmp_path / "rect.geojson"
    mask_to_geojson(mask, out)
    data = json.loads(out.read_text())
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) == 1


def test_feature_uses_polygon_geometry(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    out = tmp_path / "rect.geojson"
    mask_to_geojson(mask, out)
    data = json.loads(out.read_text())
    assert data["features"][0]["geometry"]["type"] == "Polygon"


def test_feature_name_is_str_label(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 7
    out = tmp_path / "rect.geojson"
    mask_to_geojson(mask, out)
    data = json.loads(out.read_text())
    assert data["features"][0]["properties"]["name"] == "7"


def test_disjoint_pieces_become_multipolygon(tmp_path):
    """Two disjoint blocks of the same label produce one MultiPolygon feature."""
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    mask[10:13, 10:13] = 1  # disjoint
    out = tmp_path / "mp.geojson"
    mask_to_geojson(mask, out)
    data = json.loads(out.read_text())
    assert len(data["features"]) == 1
    assert data["features"][0]["geometry"]["type"] == "MultiPolygon"


# ─── classification metadata ─────────────────────────────────────────────────

def test_annotation_dict_sets_classification_name(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    out = tmp_path / "labeled.geojson"
    mask_to_geojson(mask, out, annotation_dict={1: "Tumor"})
    data = json.loads(out.read_text())
    assert data["features"][0]["properties"]["classification"]["name"] == "Tumor"


def test_color_dict_overrides_auto_color(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    out = tmp_path / "colored.geojson"
    mask_to_geojson(
        mask, out,
        annotation_dict={1: "Tumor"},
        color_dict={"Tumor": [10, 20, 30]},
    )
    data = json.loads(out.read_text())
    assert data["features"][0]["properties"]["classification"]["color"] == [10, 20, 30]


def test_unknown_label_classified_as_unknown(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 3:7] = 1
    out = tmp_path / "unk.geojson"
    mask_to_geojson(mask, out)  # no annotation_dict
    data = json.loads(out.read_text())
    assert data["features"][0]["properties"]["classification"]["name"] == "Unknown"


# ─── simplify_tolerance ──────────────────────────────────────────────────────

def test_simplify_tolerance_reduces_vertices(tmp_path):
    """Higher tolerance → fewer polygon vertices."""
    rng = np.random.default_rng(0)
    # Use a larger image with a complex jagged blob so simplify has something
    # meaningful to remove.
    mask = np.zeros((200, 200), dtype=np.uint8)
    cy, cx = 100, 100
    for r in range(200):
        for c in range(200):
            radius_sq = 60 ** 2 + rng.integers(-1000, 1000)
            if (r - cy) ** 2 + (c - cx) ** 2 < radius_sq:
                mask[r, c] = 1

    def _count_vertices(path):
        data = json.loads(path.read_text())
        total = 0
        for f in data["features"]:
            geom = f["geometry"]
            polys = (geom["coordinates"]
                     if geom["type"] == "Polygon"
                     else [ring for poly in geom["coordinates"] for ring in poly])
            for ring in polys:
                total += len(ring)
        return total

    raw_path = tmp_path / "raw.geojson"
    simp_path = tmp_path / "simp.geojson"
    mask_to_geojson(mask, raw_path)
    mask_to_geojson(mask, simp_path, simplify_tolerance=3.0)

    assert _count_vertices(simp_path) < _count_vertices(raw_path)


# ─── round-trip ──────────────────────────────────────────────────────────────

def test_round_trip_int_names_polygon_only_false(tmp_path):
    """mask -> geojson -> mask preserves labels when polygon_only=False."""
    mask = np.zeros((50, 50), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask[20:30, 20:30] = 2
    mask[35:45, 5:15] = 3
    out = tmp_path / "rt.geojson"
    mask_to_geojson(mask, out, annotation_dict={1: "A", 2: "B", 3: "C"})
    back = geojson_to_mask(out, shape=mask.shape, polygon_only=False)
    np.testing.assert_array_equal(back, mask)


def test_round_trip_with_multipolygon(tmp_path):
    """Disjoint pieces produce MultiPolygon; round-trip requires polygon_only=False."""
    mask = np.zeros((50, 50), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask[5:15, 30:40] = 1  # disjoint, same label → MultiPolygon
    out = tmp_path / "mp.geojson"
    mask_to_geojson(mask, out)
    back = geojson_to_mask(out, shape=mask.shape, polygon_only=False)
    np.testing.assert_array_equal(back, mask)


# ─── label_by="name" default behavior ────────────────────────────────────────

def test_default_int_names_use_int_labels(data_dir):
    """multilabel.geojson has names '1'..'4'; default returns labels 1..4."""
    mask = geojson_to_mask(data_dir / "multilabel.geojson",
                            shape=(200, 200), polygon_only=False)
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2, 3, 4]


def test_default_non_int_names_fall_back_to_positional(data_dir):
    """multilabel_qupath_export.geojson has no names → polygon_1..polygon_5
    → not all int → labels are positional 1..5."""
    mask = geojson_to_mask(data_dir / "multilabel_qupath_export.geojson",
                            shape=(200, 200))
    assert sorted(np.unique(mask).tolist()) == [0, 1, 2, 3, 4, 5]


def test_label_dict_subset_skips_unlisted(data_dir):
    """label_dict naming only some polygons — others are skipped."""
    mask = geojson_to_mask(
        data_dir / "multilabel_qupath_export.geojson",
        shape=(200, 200),
        label_dict={"polygon_1": 100, "polygon_3": 200},
    )
    assert sorted(np.unique(mask).tolist()) == [0, 100, 200]


# ─── label_by="classification" ───────────────────────────────────────────────

def test_classification_mode_uses_class_names(tmp_path):
    mask = np.zeros((50, 50), dtype=np.int32)
    mask[5:15, 5:15] = 1
    mask[20:30, 20:30] = 2
    out = tmp_path / "cls.geojson"
    mask_to_geojson(mask, out, annotation_dict={1: "Tumor", 2: "Stroma"})
    back = geojson_to_mask(
        out, shape=mask.shape,
        label_by="classification",
        label_dict={"Tumor": 10, "Stroma": 20},
    )
    expected = np.zeros_like(mask)
    expected[mask == 1] = 10
    expected[mask == 2] = 20
    np.testing.assert_array_equal(back, expected)


def test_classification_mode_requires_label_dict(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    out = tmp_path / "x.geojson"
    mask_to_geojson(mask, out, annotation_dict={1: "Tumor"})
    with pytest.raises(ValueError, match="label_dict"):
        geojson_to_mask(out, shape=(10, 10), label_by="classification")


def test_invalid_label_by_raises(data_dir):
    with pytest.raises(ValueError, match="label_by"):
        geojson_to_mask(data_dir / "multilabel.geojson", shape=(200, 200),
                        label_by="bogus")


# ─── output dtype ────────────────────────────────────────────────────────────

def test_dtype_uint8_for_small_labels(tmp_path):
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 5
    out = tmp_path / "small.geojson"
    mask_to_geojson(mask, out)
    back = geojson_to_mask(out, shape=mask.shape, polygon_only=False)
    assert back.dtype == np.uint8


def test_dtype_int32_for_large_labels(tmp_path):
    mask = np.zeros((10, 10), dtype=np.int32)
    mask[2:5, 2:5] = 1000  # > 255 → must be int32
    out = tmp_path / "big.geojson"
    mask_to_geojson(mask, out)
    back = geojson_to_mask(out, shape=mask.shape, polygon_only=False)
    assert back.dtype == np.int32


# ─── empty / edge cases ──────────────────────────────────────────────────────

def test_empty_geojson_returns_zero_mask(tmp_path):
    (tmp_path / "empty.geojson").write_text(
        '{"type":"FeatureCollection","features":[]}')
    back = geojson_to_mask(tmp_path / "empty.geojson", shape=(10, 10))
    assert back.shape == (10, 10)
    assert back.sum() == 0


# ─── input validation on mask_to_geojson ────────────────────────────────────

def test_mask_to_geojson_rejects_3d_mask(tmp_path):
    """A 3D mask is almost certainly an error; raise instead of silently flattening."""
    mask = np.zeros((3, 10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="2D"):
        mask_to_geojson(mask, tmp_path / "x.geojson")


def test_mask_to_geojson_rejects_float_mask(tmp_path):
    """A float mask is ambiguous (truncation surprise); raise."""
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 2:5] = 1.5
    with pytest.raises(TypeError, match="integer"):
        mask_to_geojson(mask, tmp_path / "x.geojson")


def test_mask_to_geojson_accepts_bool_mask(tmp_path):
    """A bool mask works (treated as 0/1 binary)."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    out = tmp_path / "bool.geojson"
    mask_to_geojson(mask, out)  # should not raise
    data = json.loads(out.read_text())
    assert len(data["features"]) == 1
    assert data["features"][0]["properties"]["name"] == "1"
