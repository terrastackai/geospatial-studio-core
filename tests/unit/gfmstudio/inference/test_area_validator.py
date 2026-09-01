import pytest

from gfmstudio.inference.v2.area_validator import (
    calculate_bbox_area_sq_km,
    calculate_polygon_area_sq_km,
    calculate_total_area_sq_km,
    validate_inference_area,
)


class TestBBoxAreaCalculation:
    """Test bounding box area calculations."""

    def test_small_bbox(self):
        """Test area calculation for a small bounding box."""
        # Small area around Nairobi (approximately 1km x 1km)
        bbox = [36.8, -1.3, 36.81, -1.29]
        area = calculate_bbox_area_sq_km(bbox)

        # Should be approximately 1 km²
        assert 0.5 < area < 2.0, f"Expected ~1 km², got {area} km²"

    def test_large_bbox(self):
        """Test area calculation for a large bounding box."""
        # Approximate bounding box of Kenya
        bbox = [33.9, -4.7, 41.9, 5.5]
        area = calculate_bbox_area_sq_km(bbox)

        # Bounding box will be larger due to rectangular shape (~1,005,000 km²)
        assert 900000 < area < 1100000, f"Expected ~900,000-1,100,000 km², got {area} km²"

    def test_invalid_bbox(self):
        """Test that invalid bbox raises error."""
        with pytest.raises(ValueError):
            calculate_bbox_area_sq_km([36.8, -1.3, 36.81])  # Only 3 coordinates


class TestPolygonAreaCalculation:
    """Test polygon area calculations."""

    def test_simple_polygon(self):
        """Test area calculation for a simple polygon."""
        # Square polygon approximately 1km x 1km
        polygon = [
            [36.8, -1.3],
            [36.81, -1.3],
            [36.81, -1.29],
            [36.8, -1.29],
            [36.8, -1.3],  # Close the polygon
        ]
        area = calculate_polygon_area_sq_km(polygon)

        # Should be approximately 1 km²
        assert 0.5 < area < 2.0, f"Expected ~1 km², got {area} km²"

    def test_geojson_polygon(self):
        """Test area calculation for GeoJSON format polygon."""
        polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [36.8, -1.3],
                    [36.81, -1.3],
                    [36.81, -1.29],
                    [36.8, -1.29],
                    [36.8, -1.3],
                ]
            ],
        }
        area = calculate_polygon_area_sq_km(polygon)

        assert 0.5 < area < 2.0, f"Expected ~1 km², got {area} km²"

    def test_invalid_polygon(self):
        """Test that invalid polygon returns 0."""
        # Polygon with less than 3 points
        polygon = [[36.8, -1.3], [36.81, -1.3]]
        area = calculate_polygon_area_sq_km(polygon)

        assert area == 0.0


class TestTotalAreaCalculation:
    """Test total area calculation with multiple geometries."""

    def test_multiple_bboxes(self):
        """Test area calculation for multiple bounding boxes."""
        bboxes = [
            [36.8, -1.3, 36.81, -1.29],  # ~1 km²
            [36.82, -1.3, 36.83, -1.29],  # ~1 km²
        ]
        area = calculate_total_area_sq_km(bboxes=bboxes)

        # Should be approximately 2 km² (non-overlapping)
        assert 1.5 < area < 3.0, f"Expected ~2 km², got {area} km²"

    def test_overlapping_bboxes(self):
        """Test that overlapping areas are handled correctly."""
        bboxes = [
            [36.8, -1.3, 36.81, -1.29],
            [36.805, -1.3, 36.815, -1.29],  # Overlaps with first
        ]
        area = calculate_total_area_sq_km(bboxes=bboxes)

        # Should be less than 2 km² due to overlap
        assert 1.0 < area < 2.0, f"Expected ~1.5 km² (with overlap), got {area} km²"

    def test_mixed_geometries(self):
        """Test area calculation with both bboxes and polygons."""
        bboxes = [[36.8, -1.3, 36.81, -1.29]]
        polygons = [
            [
                [36.82, -1.3],
                [36.83, -1.3],
                [36.83, -1.29],
                [36.82, -1.29],
                [36.82, -1.3],
            ]
        ]
        area = calculate_total_area_sq_km(bboxes=bboxes, polygons=polygons)

        # Should be approximately 2 km²
        assert 1.5 < area < 3.0, f"Expected ~2 km², got {area} km²"


class TestAreaValidation:
    """Test inference area validation."""

    def test_validation_disabled(self):
        """Test that validation is disabled when max_area is None or 0."""
        bbox = [33.9, -4.7, 41.9, 5.5]  # Large area

        # With None
        is_valid, area, error = validate_inference_area(
            bboxes=[bbox], max_area_sq_km=None
        )
        assert is_valid is True
        assert error is None

        # With 0
        is_valid, area, error = validate_inference_area(bboxes=[bbox], max_area_sq_km=0)
        assert is_valid is True
        assert error is None

    def test_area_within_limit(self):
        """Test that small areas pass validation."""
        bbox = [36.8, -1.3, 36.81, -1.29]  # ~1 km²

        is_valid, area, error = validate_inference_area(
            bboxes=[bbox], max_area_sq_km=100.0  # 100 km² limit
        )

        assert is_valid is True
        assert error is None
        assert area < 100.0

    def test_area_exceeds_limit(self):
        """Test that large areas fail validation."""
        bbox = [33.9, -4.7, 41.9, 5.5]  # Large area (Kenya-sized)

        is_valid, area, error = validate_inference_area(
            bboxes=[bbox], max_area_sq_km=100.0  # 100 km² limit
        )

        assert is_valid is False
        assert error is not None
        assert "exceeds the maximum allowed area" in error
        assert area > 100.0

    def test_default_kenya_limit(self):
        """Test with default limit (0.125 of Kenya = ~72,546 km²)."""
        # Small area should pass
        small_bbox = [36.8, -1.3, 36.81, -1.29]
        is_valid, _, _ = validate_inference_area(
            bboxes=[small_bbox], max_area_sq_km=72546.0
        )
        assert is_valid is True

        # Very large area should fail
        large_bbox = [33.9, -4.7, 41.9, 5.5]
        is_valid, area, error = validate_inference_area(
            bboxes=[large_bbox], max_area_sq_km=72546.0
        )
        assert is_valid is False
        assert error is not None
