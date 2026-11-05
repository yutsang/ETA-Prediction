"""Geospatial utilities for HK ETA system."""
import math
from typing import Tuple


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return great-circle distance in kilometres between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def parse_coord(coord_str: str) -> Tuple[float, float]:
    """Parse a 'lat, lon' string into (lat, lon) floats. Returns (None, None) on failure."""
    try:
        parts = coord_str.split(",")
        return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None, None


# HK district bounding boxes for rough classification
_HK_DISTRICTS = {
    "Hong Kong Island": (22.24, 22.30, 114.12, 114.22),
    "Kowloon":          (22.30, 22.35, 114.15, 114.22),
    "New Territories":  (22.35, 22.55, 113.83, 114.32),
    "Lantau":           (22.18, 22.32, 113.83, 114.05),
}


def classify_district(lat: float, lon: float) -> str:
    """Return rough HK district name for a coordinate pair."""
    if lat is None or lon is None:
        return "Unknown"
    for name, (lat_min, lat_max, lon_min, lon_max) in _HK_DISTRICTS.items():
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            return name
    return "Other"
