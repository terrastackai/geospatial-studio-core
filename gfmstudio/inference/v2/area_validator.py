import math
from typing import List, Optional, Tuple

from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def calculate_bbox_area_sq_km(bbox: List[float]) -> float:
    """
    Calculate the area of a bounding box in square kilometers.

    Args:
        bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]

    Returns:
        Area in square kilometers
    """
    if len(bbox) != 4:
        raise ValueError("Bounding box must have exactly 4 coordinates")

    min_lon, min_lat, max_lon, max_lat = bbox

    # Create a box polygon
    bbox_polygon = box(min_lon, min_lat, max_lon, max_lat)

    # Calculate area using geodesic approximation
    # For small areas, we can use a simple approximation
    # For more accuracy, we'd need to use a proper geodesic library

    # Average latitude for the area
    avg_lat = (min_lat + max_lat) / 2

    # Degrees to kilometers conversion
    # 1 degree latitude ≈ 111 km
    # 1 degree longitude ≈ 111 km * cos(latitude)
    lat_km_per_degree = 111.0
    lon_km_per_degree = 111.0 * math.cos(math.radians(avg_lat))

    # Calculate dimensions in km
    width_km = (max_lon - min_lon) * lon_km_per_degree
    height_km = (max_lat - min_lat) * lat_km_per_degree

    # Area in square kilometers
    area_sq_km = width_km * height_km

    return abs(area_sq_km)


def calculate_polygon_area_sq_km(polygon_coords: List) -> float:
    """
    Calculate the area of a polygon in square kilometers.

    Args:
        polygon_coords: Polygon coordinates in GeoJSON format

    Returns:
        Area in square kilometers
    """
    try:
        # Handle different polygon formats
        if isinstance(polygon_coords, dict):
            # GeoJSON format
            if polygon_coords.get("type") == "Polygon":
                coords = polygon_coords.get("coordinates", [[]])[0]
            else:
                coords = polygon_coords.get("coordinates", [])
        else:
            coords = polygon_coords

        if not coords or len(coords) < 3:
            return 0.0

        # Create shapely polygon
        poly = Polygon(coords)

        # Get bounding box for approximation
        bounds = poly.bounds  # (minx, miny, maxx, maxy)

        # Calculate area using bounding box approximation
        # This is a rough estimate; for precise calculations, use a geodesic library
        avg_lat = (bounds[1] + bounds[3]) / 2
        lat_km_per_degree = 111.0
        lon_km_per_degree = 111.0 * math.cos(math.radians(avg_lat))

        # Get area in square degrees
        area_sq_degrees = poly.area

        # Convert to square kilometers
        area_sq_km = area_sq_degrees * lat_km_per_degree * lon_km_per_degree

        return abs(area_sq_km)
    except Exception as e:
        # If calculation fails, return 0 to allow the request
        # (better to be permissive than block valid requests)
        return 0.0


def calculate_total_area_sq_km(
    bboxes: Optional[List[List[float]]] = None, polygons: Optional[List] = None
) -> float:
    """
    Calculate the total area covered by bounding boxes and polygons.
    Handles overlapping areas by using union.

    Args:
        bboxes: List of bounding boxes
        polygons: List of polygons

    Returns:
        Total area in square kilometers
    """
    geometries = []

    # Add bounding boxes
    if bboxes:
        for bbox in bboxes:
            if bbox and len(bbox) == 4:
                try:
                    geometries.append(box(bbox[0], bbox[1], bbox[2], bbox[3]))
                except Exception:
                    continue

    # Add polygons
    if polygons:
        for poly_data in polygons:
            try:
                if isinstance(poly_data, dict):
                    if poly_data.get("type") == "Polygon":
                        coords = poly_data.get("coordinates", [[]])[0]
                    else:
                        coords = poly_data.get("coordinates", [])
                else:
                    coords = poly_data

                if coords and len(coords) >= 3:
                    geometries.append(Polygon(coords))
            except Exception:
                continue

    if not geometries:
        return 0.0

    # Union all geometries to handle overlaps
    try:
        union_geom = unary_union(geometries)
        bounds = union_geom.bounds  # (minx, miny, maxx, maxy)

        # Calculate area using geodesic approximation
        avg_lat = (bounds[1] + bounds[3]) / 2
        lat_km_per_degree = 111.0
        lon_km_per_degree = 111.0 * math.cos(math.radians(avg_lat))

        # Get area in square degrees
        area_sq_degrees = union_geom.area

        # Convert to square kilometers
        area_sq_km = area_sq_degrees * lat_km_per_degree * lon_km_per_degree

        return abs(area_sq_km)
    except Exception:
        # If union fails, sum individual areas (may overestimate due to overlaps)
        total_area = 0.0
        if bboxes:
            for bbox in bboxes:
                if bbox and len(bbox) == 4:
                    try:
                        total_area += calculate_bbox_area_sq_km(bbox)
                    except Exception:
                        continue

        if polygons:
            for poly in polygons:
                try:
                    total_area += calculate_polygon_area_sq_km(poly)
                except Exception:
                    continue

        return total_area


def validate_inference_area(
    bboxes: Optional[List[List[float]]] = None,
    polygons: Optional[List] = None,
    max_area_sq_km: Optional[float] = None,
) -> Tuple[bool, float, Optional[str]]:
    """
    Validate that the inference area doesn't exceed the maximum allowed area.

    Args:
        bboxes: List of bounding boxes
        polygons: List of polygons
        max_area_sq_km: Maximum allowed area in square kilometers.
                       If None or 0, validation is disabled.

    Returns:
        Tuple of (is_valid, calculated_area_sq_km, error_message)
    """
    # If validation is disabled, always return valid
    if max_area_sq_km is None or max_area_sq_km <= 0:
        return (True, 0.0, None)

    # Calculate total area
    total_area = calculate_total_area_sq_km(bboxes=bboxes, polygons=polygons)

    # Check if area exceeds limit
    if total_area > max_area_sq_km:
        error_msg = (
            f"Inference area ({total_area:.2f} km²) exceeds the maximum allowed "
            f"area of {max_area_sq_km:.2f} km². Please reduce the spatial extent "
            f"of your inference request."
        )
        return (False, total_area, error_msg)

    return (True, total_area, None)
