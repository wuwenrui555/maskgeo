"""maskgeo — convert between segmentation masks and pixel-edge polygons (GeoJSON)."""

from maskgeo.convert import geojson_to_mask, mask_to_geojson
from maskgeo.processor import GeojsonProcessor, PolygonProcessor

__version__ = "1.0.0"

__all__ = ["GeojsonProcessor", "PolygonProcessor", "geojson_to_mask", "mask_to_geojson"]
