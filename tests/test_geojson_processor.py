"""Tests for GeojsonProcessor — load, sanitize, name-normalize, classify, crop."""
import json

import geopandas as gpd
import matplotlib
import numpy as np
import pytest
from shapely.geometry import LineString, MultiPolygon, Polygon

matplotlib.use("Agg")  # non-interactive backend for CI
import matplotlib.pyplot as plt  # noqa: E402

from maskgeo.processor import GeojsonProcessor  # noqa: E402


# ─── helpers ─────────────────────────────────────────────────────────────────

def _gdf_from_features(features: list[dict]) -> gpd.GeoDataFrame:
    """Helper: build a GeoDataFrame from a list of GeoJSON-like feature dicts."""
    return gpd.GeoDataFrame.from_features(features)


def _poly_feature(coords, name=None):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def _multipoly_feature(rings, name=None):
    feat = {
        "type": "Feature",
        "geometry": {"type": "MultiPolygon",
            "coordinates": [[r] for r in rings]},
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def _line_feature(coords, name=None):
    feat = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def _two_polygon_gdf():
    feats = [
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
         "properties": {"name": "A"}},
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]},
         "properties": {"name": "B"}},
    ]
    return _gdf_from_features(feats)


# ─── from_path basic shape ───────────────────────────────────────────────────

def test_from_path_qupath_export(data_dir):
    """The real QuPath export (no name, no classification) loads cleanly."""
    gp = GeojsonProcessor.from_path(data_dir / "multilabel_qupath_export.geojson")
    assert len(gp.gdf) == 5


def test_from_path_loads_multilabel_geojson(data_dir):
    """The mask_to_geojson-generated fixture loads with names '1'..'4'."""
    gp = GeojsonProcessor.from_path(data_dir / "multilabel.geojson", polygon_only=False)
    assert sorted(gp.gdf.index.tolist()) == ["1", "2", "3", "4"]


# ─── name auto-fill ──────────────────────────────────────────────────────────

def test_from_path_auto_fills_missing_names(data_dir):
    """Features with no `properties.name` get polygon_1..polygon_N."""
    gp = GeojsonProcessor.from_path(data_dir / "multilabel_qupath_export.geojson")
    assert gp.gdf.index.tolist() == [
        "polygon_1", "polygon_2", "polygon_3", "polygon_4", "polygon_5",
    ]


def test_name_prefix_constant():
    assert GeojsonProcessor.NAME_PREFIX == "polygon_"


# ─── name dedup ──────────────────────────────────────────────────────────────

def test_init_dedups_repeated_names():
    feats = [
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
         "properties": {"name": "Tumor"}},
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]},
         "properties": {"name": "Tumor"}},
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[4, 0], [5, 0], [5, 1], [4, 1], [4, 0]]]},
         "properties": {"name": "Stroma"}},
    ]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert sorted(gp.gdf.index.tolist()) == ["Stroma", "Tumor_1", "Tumor_2"]


# ─── geometry sanitization ───────────────────────────────────────────────────

def test_polygon_only_default_drops_balanced_multipolygon():
    """polygon_only=True (default), MultiPolygon with comparable pieces dropped."""
    a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    b = [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]
    feats = [_multipoly_feature([a, b], name="balanced")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert len(gp.gdf) == 0
    assert len(gp.skipped) == 1
    assert gp.skipped[0]["name"] == "balanced"
    assert "MultiPolygon" in gp.skipped[0]["reason"]


def test_polygon_only_default_keeps_dominant_multipolygon():
    """Dominant MultiPolygon (area ratio >= 100) keeps the largest piece."""
    big = [[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]
    tiny = [[200, 200], [201, 200], [201, 201], [200, 201], [200, 200]]
    feats = [_multipoly_feature([big, tiny], name="dom")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert len(gp.gdf) == 1
    assert isinstance(gp.gdf.geometry.iloc[0], Polygon)


def test_polygon_only_false_keeps_multipolygon_intact():
    """polygon_only=False keeps MultiPolygon as MultiPolygon."""
    a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    b = [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]
    feats = [_multipoly_feature([a, b], name="mp")]
    gp = GeojsonProcessor(_gdf_from_features(feats), polygon_only=False)
    assert len(gp.gdf) == 1
    assert isinstance(gp.gdf.geometry.iloc[0], MultiPolygon)
    assert len(gp.skipped) == 0


def test_linestring_close_endpoints_kept_as_polygon():
    coords = [[0, 0], [10, 0], [10, 10], [0.1, 0.1]]  # endpoints close
    feats = [_line_feature(coords, name="line")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert len(gp.gdf) == 1
    assert isinstance(gp.gdf.geometry.iloc[0], Polygon)


def test_linestring_far_endpoints_dropped():
    coords = [[0, 0], [100, 0], [100, 100]]
    feats = [_line_feature(coords, name="open")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert len(gp.gdf) == 0
    assert gp.skipped[0]["name"] == "open"
    assert "LineString" in gp.skipped[0]["reason"]


# ─── inspection ──────────────────────────────────────────────────────────────

def test_skipped_attribute_format():
    """gp.skipped is a list of dicts with index/name/reason."""
    coords = [[0, 0], [100, 0], [100, 100]]
    feats = [_line_feature(coords, name="open")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    assert isinstance(gp.skipped, list)
    s = gp.skipped[0]
    assert set(s.keys()) >= {"index", "name", "reason"}
    assert isinstance(s["index"], int)
    assert isinstance(s["name"], str)
    assert isinstance(s["reason"], str)


def test_gdf_raw_preserves_original():
    """gp.gdf_raw is the unmodified input, gp.gdf is sanitized."""
    a = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    b = [[20, 0], [30, 0], [30, 10], [20, 10], [20, 0]]
    feats = [_multipoly_feature([a, b], name="mp")]
    gp = GeojsonProcessor(_gdf_from_features(feats))  # default polygon_only=True
    assert len(gp.gdf_raw) == 1
    assert len(gp.gdf) == 0


# ─── from_text ───────────────────────────────────────────────────────────────

def test_from_text_loads_geojson_string(data_dir):
    text = (data_dir / "multilabel_qupath_export.geojson").read_text()
    gp = GeojsonProcessor.from_text(text)
    assert len(gp.gdf) == 5


# ─── update_classification ───────────────────────────────────────────────────

def test_update_classification_sets_name():
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification({"A": "Tumor", "B": "Stroma"})
    a_cls = json.loads(gp.gdf.loc["A", "classification"])
    b_cls = json.loads(gp.gdf.loc["B", "classification"])
    assert a_cls["name"] == "Tumor"
    assert b_cls["name"] == "Stroma"


def test_update_classification_auto_color_when_color_dict_none():
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification({"A": "Tumor", "B": "Stroma"})
    a_cls = json.loads(gp.gdf.loc["A", "classification"])
    b_cls = json.loads(gp.gdf.loc["B", "classification"])
    assert a_cls["color"] != b_cls["color"]
    assert all(0 <= c <= 255 for c in a_cls["color"])


def test_update_classification_explicit_color_overrides():
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification(
        {"A": "Tumor", "B": "Stroma"},
        color_dict={"Tumor": [11, 22, 33]},
    )
    a_cls = json.loads(gp.gdf.loc["A", "classification"])
    assert a_cls["color"] == [11, 22, 33]


def test_update_classification_unmapped_polygon_keeps_classification():
    gp = GeojsonProcessor(_two_polygon_gdf())
    original_b = gp.gdf.loc["B", "classification"]
    gp.update_classification({"A": "Tumor"})  # B not in name_dict
    assert gp.gdf.loc["B", "classification"] == original_b


def test_update_classification_warns_on_unknown_name(caplog):
    """Names in name_dict that don't exist in gp.gdf.index get a warning."""
    import logging
    gp = GeojsonProcessor(_two_polygon_gdf())
    with caplog.at_level(logging.WARNING, logger="maskgeo.processor"):
        gp.update_classification({"A": "Tumor", "Typo": "Stroma"})
    assert any("Typo" in rec.message for rec in caplog.records)


# ─── output_geojson round-trip ───────────────────────────────────────────────

def test_output_geojson_roundtrip(tmp_path):
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification({"A": "Tumor", "B": "Stroma"})
    out = tmp_path / "out.geojson"
    gp.output_geojson(out)
    reloaded = GeojsonProcessor.from_path(out)
    assert sorted(reloaded.gdf.index.tolist()) == ["A", "B"]
    a_cls = json.loads(reloaded.gdf.loc["A", "classification"])
    assert a_cls["name"] == "Tumor"


# ─── plotting ────────────────────────────────────────────────────────────────

def test_plot_classification_returns_figure():
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification({"A": "Tumor", "B": "Stroma"})
    fig = gp.plot_classification()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_classification_uses_existing_axis():
    gp = GeojsonProcessor(_two_polygon_gdf())
    gp.update_classification({"A": "Tumor", "B": "Stroma"})
    fig, ax = plt.subplots()
    fig2 = gp.plot_classification(ax=ax)
    assert fig2 is fig
    plt.close(fig)


def test_plot_name_returns_figure():
    gp = GeojsonProcessor(_two_polygon_gdf())
    fig = gp.plot_name()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_name_text_disabled():
    gp = GeojsonProcessor(_two_polygon_gdf())
    fig = gp.plot_name(text=False)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ─── crop_image ──────────────────────────────────────────────────────────────

def test_crop_image_2d_array():
    """2D ndarray input → yields (name, 2D cropped array) per polygon."""
    img = np.arange(400, dtype=np.uint8).reshape(20, 20)
    feats = [
        _poly_feature([[2, 3], [5, 3], [5, 6], [2, 6], [2, 3]], name="r1"),
        _poly_feature([[10, 12], [13, 12], [13, 15], [10, 15], [10, 12]], name="r2"),
    ]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    results = dict(gp.crop_image(img))
    assert set(results.keys()) == {"r1", "r2"}
    assert results["r1"].shape == (3, 3)
    assert results["r2"].shape == (3, 3)


def test_crop_image_3d_cyx():
    img = np.zeros((4, 20, 20), dtype=np.uint8)
    img[2, 3:6, 2:5] = 99
    feats = [_poly_feature([[2, 3], [5, 3], [5, 6], [2, 6], [2, 3]], name="r")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    results = dict(gp.crop_image(img, dim_order="CYX"))
    assert results["r"].shape == (4, 3, 3)
    assert (results["r"][2] == 99).all()


def test_crop_image_dict_input_yields_dict():
    """dict[str, ndarray] input → yields (name, dict[channel, 2D ndarray])."""
    img_dict = {
        "DAPI": np.arange(400, dtype=np.uint16).reshape(20, 20),
        "CD45": np.arange(400, 800, dtype=np.uint16).reshape(20, 20),
    }
    feats = [_poly_feature([[2, 3], [5, 3], [5, 6], [2, 6], [2, 3]], name="r")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    results = dict(gp.crop_image(img_dict))
    assert set(results["r"].keys()) == {"DAPI", "CD45"}
    assert results["r"]["DAPI"].shape == (3, 3)
    assert results["r"]["CD45"].shape == (3, 3)


def test_crop_image_invalid_dim_order():
    img = np.zeros((3, 10, 10), dtype=np.uint8)
    feats = [_poly_feature([[2, 3], [5, 3], [5, 6], [2, 6], [2, 3]], name="r")]
    gp = GeojsonProcessor(_gdf_from_features(feats))
    with pytest.raises(ValueError, match="dim_order"):
        list(gp.crop_image(img, dim_order="ZZZ"))


def test_crop_image_handles_multipolygon():
    """polygon_only=False keeps MultiPolygon; crop_image must work for them too."""
    a = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]
    b = [[10, 10], [12, 10], [12, 12], [10, 12], [10, 10]]
    feats = [_multipoly_feature([a, b], name="mp")]
    gp = GeojsonProcessor(_gdf_from_features(feats), polygon_only=False)
    img = np.full((20, 20), 99, dtype=np.uint8)
    results = dict(gp.crop_image(img))
    assert "mp" in results
    cropped = results["mp"]
    # bbox of the MultiPolygon spans (2,2)..(12,12) → cropped shape (10, 10).
    assert cropped.shape == (10, 10)
    # Both blocks should be present (value 99); outside should be 0.
    assert (cropped[0:2, 0:2] == 99).all()  # piece a (rows 2..3, cols 2..3 in img)
    assert (cropped[8:10, 8:10] == 99).all()  # piece b (rows 10..11, cols 10..11)
    assert (cropped[5, 5] == 0)  # gap between pieces
