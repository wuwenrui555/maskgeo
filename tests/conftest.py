"""Shared pytest fixtures and helpers for maskgeo tests.

Fixture files in tests/data/ were captured during the 2026-04-26 brainstorming
session and validated against QuPath. See docs/superpowers/specs/.
"""
from pathlib import Path

import geopandas as gpd
import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def data_dir() -> Path:
    return DATA_DIR


# ─── feature-builder helpers (used across test_geojson_processor.py etc.) ─────


def gdf_from_features(features: list[dict]) -> gpd.GeoDataFrame:
    """Build a GeoDataFrame from a list of GeoJSON-like feature dicts."""
    return gpd.GeoDataFrame.from_features(features)


def poly_feature(coords, name=None):
    feat = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def multipoly_feature(rings, name=None):
    feat = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [[r] for r in rings],
        },
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def line_feature(coords, name=None):
    feat = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords},
        "properties": {},
    }
    if name is not None:
        feat["properties"]["name"] = name
    return feat


def two_polygon_gdf():
    return gdf_from_features([
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
         "properties": {"name": "A"}},
        {"type": "Feature", "geometry": {"type": "Polygon",
            "coordinates": [[[2, 0], [3, 0], [3, 1], [2, 1], [2, 0]]]},
         "properties": {"name": "B"}},
    ])
