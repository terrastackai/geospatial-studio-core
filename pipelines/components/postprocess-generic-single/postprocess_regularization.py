##########################################################################################
#
# /* MIT License
#  *
#  * ------------------------------------------
#  *
#  * Copyright (c) 2023-2025, Qiusheng Wu
#  *
#  * Permission is hereby granted, free of charge, to any person obtaining a copy
#  * of this software and associated documentation files (the "Software"), to deal
#  * in the Software without restriction, including without limitation the rights
#  * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  * copies of the Software, and to permit persons to whom the Software is
#  * furnished to do so, subject to the following conditions:
#  *
#  * The above copyright notice and this permission notice shall be included in all
#  * copies or substantial portions of the Software.
#  *
#  * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  * SOFTWARE.
#  */
#
##########################################################################################

import os
import pyproj
import rasterio
import numpy as np
import geopandas as gpd
import buildingregulariser
from rasterio import features
import matplotlib.pyplot as plt
from typing import List, Optional, Union
from shapely.geometry import shape, Polygon


def raster_to_vector(
    raster_path: str,
    output_path: Optional[str] = None,
    threshold: float = 0,
    min_area: float = 10,
    simplify_tolerance: Optional[float] = None,
    class_values: Optional[List[int]] = None,
    attribute_name: str = "class",
    unique_attribute_value: bool = False,
    output_format: str = "geojson",
    plot_result: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert a raster label mask to vector polygons.

    Args:
        raster_path (str): Path to the input raster file (e.g., GeoTIFF).
        output_path (str): Path to save the output vector file. If None, returns GeoDataFrame without saving.
        threshold (int/float): Pixel values greater than this threshold will be vectorized.
        min_area (float): Minimum polygon area in square map units to keep.
        simplify_tolerance (float): Tolerance for geometry simplification. None for no simplification.
        class_values (list): Specific pixel values to vectorize. If None, all values > threshold are vectorized.
        attribute_name (str): Name of the attribute field for the class values.
        unique_attribute_value (bool): Whether to generate unique values for each shape within a class.
        output_format (str): Format for output file - 'geojson', 'shapefile', 'gpkg'.
        plot_result (bool): Whether to plot the resulting polygons overlaid on the raster.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame containing the vectorized polygons.
    """
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the data
        data = src.read(1)

        # Get metadata
        transform = src.transform
        crs = src.crs

        # Create mask based on threshold and class values
        if class_values is not None:
            # Create a mask for each specified class value
            masks = {val: (data == val) for val in class_values}
        else:
            # Create a mask for values above threshold
            masks = {1: (data > threshold)}
            class_values = [1]  # Default class

        # Initialize list to store features
        all_features = []

        # Process each class value
        for class_val in class_values:
            mask = masks[class_val]
            shape_count = 1
            # Vectorize the mask
            for geom, value in features.shapes(mask.astype(np.uint8), mask=mask, transform=transform):
                # Convert to shapely geometry
                geom = shape(geom)

                # Skip small polygons
                if geom.area < min_area:
                    continue

                # Simplify geometry if requested
                if simplify_tolerance is not None:
                    geom = geom.simplify(simplify_tolerance)

                # Add to features list with class value
                if unique_attribute_value:
                    all_features.append({"geometry": geom, attribute_name: class_val * shape_count})
                else:
                    all_features.append({"geometry": geom, attribute_name: class_val})

                shape_count += 1

        # Create GeoDataFrame
        if all_features:
            gdf = gpd.GeoDataFrame(all_features, crs=crs)
        else:
            print("Warning: No features were extracted from the raster.")
            # Return empty GeoDataFrame with correct CRS
            gdf = gpd.GeoDataFrame([], geometry=[], crs=crs)

        # Save to file if requested
        if output_path is not None:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Save to file based on format
            if output_format.lower() == "geojson":
                gdf.to_file(output_path, driver="GeoJSON")
            elif output_format.lower() == "shapefile":
                gdf.to_file(output_path)
            elif output_format.lower() == "gpkg":
                gdf.to_file(output_path, driver="GPKG")
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            print(f"Vectorized data saved to {output_path}")

        # Plot result if requested
        if plot_result:
            fig, ax = plt.subplots(figsize=(12, 12))

            # Plot raster
            raster_img = src.read()
            if raster_img.shape[0] == 1:
                plt.imshow(raster_img[0], cmap="viridis", alpha=0.7)
            else:
                # Use first 3 bands for RGB display
                rgb = raster_img[:3].transpose(1, 2, 0)
                # Normalize for display
                rgb = np.clip(rgb / rgb.max(), 0, 1)
                plt.imshow(rgb)

            # Plot vector boundaries
            if not gdf.empty:
                gdf.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)

            plt.title("Raster with Vectorized Boundaries")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

        return gdf


def adaptive_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    simplify_tolerance: float = 0.5,
    area_threshold: float = 0.9,
    preserve_shape: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Adaptively regularizes building footprints based on their characteristics.

    This approach determines the best regularization method for each building.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons
        simplify_tolerance: Distance tolerance for simplification
        area_threshold: Minimum acceptable area ratio
        preserve_shape: Whether to preserve overall shape for complex buildings

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Analyze the overall dataset to set appropriate parameters
    if is_gdf := isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # Skip invalid geometries
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Measure building complexity
        complexity = building.length / (4 * np.sqrt(building.area))

        # Determine if the building has a clear principal direction
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Normalize angles to 0-180 range and get histogram
        norm_angles = angles % 180
        hist, bins = np.histogram(norm_angles, bins=18, range=(0, 180), weights=segment_lengths)

        # Calculate direction clarity (ratio of longest direction to total)
        direction_clarity = np.max(hist) / np.sum(hist) if np.sum(hist) > 0 else 0

        # Choose regularization method based on building characteristics
        if complexity < 1.2 and direction_clarity > 0.5:
            # Simple building with clear direction: use rotated rectangle
            bin_max = np.argmax(hist)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            dominant_angle = bin_centers[bin_max]

            # Rotate to align with coordinate system
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create bounding box in rotated space
            bounds = rotated.bounds
            rect = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            # Rotate back
            result = rotate(rect, dominant_angle, origin="centroid")

            # Quality check
            if result.area / building.area < area_threshold or result.area / building.area > (1.0 / area_threshold):
                # Too much area change, use simplified original
                result = building.simplify(simplify_tolerance, preserve_topology=True)

        else:
            # Complex building or no clear direction: preserve shape
            if preserve_shape:
                # Simplify with topology preservation
                result = building.simplify(simplify_tolerance, preserve_topology=True)
            else:
                # Fall back to convex hull for very complex shapes
                result = building.convex_hull

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
    angle_tolerance: float = 10,
    simplify_tolerance: float = 0.5,
    orthogonalize: bool = True,
    preserve_topology: bool = True,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    Regularizes building footprint polygons with multiple techniques beyond minimum
    rotated rectangles.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints
        angle_tolerance: Degrees within which angles will be regularized to 90/180 degrees
        simplify_tolerance: Distance tolerance for Douglas-Peucker simplification
        orthogonalize: Whether to enforce orthogonal angles in the final polygons
        preserve_topology: Whether to preserve topology during simplification

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely import wkt
    from shapely.affinity import rotate, translate
    from shapely.geometry import Polygon, shape

    regularized_buildings = []

    # Check if we're dealing with a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    for building in geom_objects:
        # Handle potential string representations of geometries
        if isinstance(building, str):
            try:
                # Try to parse as WKT
                building = wkt.loads(building)
            except Exception:
                print(f"Failed to parse geometry string: {building[:30]}...")
                continue

        # Ensure we have a valid geometry
        if not hasattr(building, "simplify"):
            print(f"Invalid geometry type: {type(building)}")
            continue

        # Step 1: Simplify to remove noise and small vertices
        simplified = building.simplify(simplify_tolerance, preserve_topology=preserve_topology)

        if orthogonalize:
            # Make sure we have a valid polygon with an exterior
            if not hasattr(simplified, "exterior") or simplified.exterior is None:
                print(f"Simplified geometry has no exterior: {simplified}")
                regularized_buildings.append(building)  # Use original instead
                continue

            # Step 2: Get the dominant angle to rotate building
            coords = np.array(simplified.exterior.coords)

            # Make sure we have enough coordinates for angle calculation
            if len(coords) < 3:
                print(f"Not enough coordinates for angle calculation: {len(coords)}")
                regularized_buildings.append(building)  # Use original instead
                continue

            segments = np.diff(coords, axis=0)
            angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

            # Find most common angle classes (0, 90, 180, 270 degrees)
            binned_angles = np.round(angles / 90) * 90
            dominant_angle = np.bincount(binned_angles.astype(int) % 180).argmax()

            # Step 3: Rotate to align with axes, regularize, then rotate back
            rotated = rotate(simplified, -dominant_angle, origin="centroid")

            # Step 4: Rectify coordinates to enforce right angles
            ext_coords = np.array(rotated.exterior.coords)
            rect_coords = []

            # Regularize each vertex to create orthogonal corners
            for i in range(len(ext_coords) - 1):
                rect_coords.append(ext_coords[i])

                # Check if we need to add a right-angle vertex
                angle = (
                    np.arctan2(
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 1] - ext_coords[i, 1],
                        ext_coords[(i + 1) % (len(ext_coords) - 1), 0] - ext_coords[i, 0],
                    )
                    * 180
                    / np.pi
                )

                if abs(angle % 90) > angle_tolerance and abs(angle % 90) < (90 - angle_tolerance):
                    # Add intermediate point to create right angle
                    rect_coords.append(
                        [
                            ext_coords[(i + 1) % (len(ext_coords) - 1), 0],
                            ext_coords[i, 1],
                        ]
                    )

            # Close the polygon by adding the first point again
            rect_coords.append(rect_coords[0])

            # Create regularized polygon and rotate back
            regularized = Polygon(rect_coords)
            final_building = rotate(regularized, dominant_angle, origin="centroid")
        else:
            final_building = simplified

        regularized_buildings.append(final_building)

    # If input was a GeoDataFrame, return a GeoDataFrame
    if isinstance(building_polygons, gpd.GeoDataFrame):
        return gpd.GeoDataFrame(geometry=regularized_buildings, crs=building_polygons.crs)
    else:
        return regularized_buildings


def hybrid_regularization(
    building_polygons: Union[gpd.GeoDataFrame, List[Polygon]],
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """
    A comprehensive hybrid approach to building footprint regularization.

    Applies different strategies based on building characteristics.

    Args:
        building_polygons: GeoDataFrame or list of shapely Polygons containing building footprints

    Returns:
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    from shapely.affinity import rotate
    from shapely.geometry import Polygon

    # Use minimum_rotated_rectangle instead of oriented_envelope
    try:
        from shapely.minimum_rotated_rectangle import minimum_rotated_rectangle
    except ImportError:
        # For older Shapely versions
        def minimum_rotated_rectangle(geom):
            """Calculate the minimum rotated rectangle for a geometry"""
            # For older Shapely versions, implement a simple version
            return geom.minimum_rotated_rectangle

    # Determine input type for correct return
    is_gdf = isinstance(building_polygons, gpd.GeoDataFrame)

    # Extract geometries if GeoDataFrame
    if is_gdf:
        geom_objects = building_polygons.geometry
    else:
        geom_objects = building_polygons

    results = []

    for building in geom_objects:
        # 1. Analyze building characteristics
        if not hasattr(building, "exterior") or building.is_empty:
            results.append(building)
            continue

        # Calculate shape complexity metrics
        complexity = building.length / (4 * np.sqrt(building.area))

        # Calculate dominant angle
        coords = np.array(building.exterior.coords)[:-1]
        segments = np.diff(np.vstack([coords, coords[0]]), axis=0)
        segment_lengths = np.sqrt(segments[:, 0] ** 2 + segments[:, 1] ** 2)
        segment_angles = np.arctan2(segments[:, 1], segments[:, 0]) * 180 / np.pi

        # Weight angles by segment length
        hist, bins = np.histogram(segment_angles % 180, bins=36, range=(0, 180), weights=segment_lengths)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        dominant_angle = bin_centers[np.argmax(hist)]

        # Check if building is close to orthogonal
        is_orthogonal = min(dominant_angle % 45, 45 - (dominant_angle % 45)) < 5

        # 2. Apply appropriate regularization strategy
        if complexity > 1.5:
            # Complex buildings: use minimum rotated rectangle
            result = minimum_rotated_rectangle(building)
        elif is_orthogonal:
            # Near-orthogonal buildings: orthogonalize in place
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Create orthogonal hull in rotated space
            bounds = rotated.bounds
            ortho_hull = Polygon(
                [
                    (bounds[0], bounds[1]),
                    (bounds[2], bounds[1]),
                    (bounds[2], bounds[3]),
                    (bounds[0], bounds[3]),
                ]
            )

            result = rotate(ortho_hull, dominant_angle, origin="centroid")
        else:
            # Diagonal buildings: use custom approach for diagonal buildings
            # Rotate to align with axes
            rotated = rotate(building, -dominant_angle, origin="centroid")

            # Simplify in rotated space
            simplified = rotated.simplify(0.3, preserve_topology=True)

            # Get the bounds in rotated space
            bounds = simplified.bounds
            min_x, min_y, max_x, max_y = bounds

            # Create a rectangular hull in rotated space
            rect_poly = Polygon([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

            # Rotate back to original orientation
            result = rotate(rect_poly, dominant_angle, origin="centroid")

        results.append(result)

    # Return in same format as input
    if is_gdf:
        return gpd.GeoDataFrame(geometry=results, crs=building_polygons.crs)
    else:
        return results


def buildingregulariser_regularize_geodataframe(
    geodataframe: gpd.GeoDataFrame,
    parallel_threshold: float = 1.0,
    target_crs: Optional[Union[str, pyproj.CRS]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 0,
    include_metadata: bool = False,
    neighbor_alignment: bool = False,
    neighbor_search_distance: float = 100.0,
    neighbor_max_rotation: float = 10,
):
    """
    Regularizes polygon geometries in a GeoDataFrame by aligning edges.

    Aligns edges to be parallel or perpendicular (optionally also 45 degrees)
    to their main direction. Handles reprojection, initial simplification,
    regularization, geometry cleanup, and parallel processing.

    Parameters:
    -----------
    geodataframe : geopandas.GeoDataFrame
        Input GeoDataFrame with polygon or multipolygon geometries.
    parallel_threshold : float, optional
        Distance threshold for merging nearly parallel adjacent edges during
        regularization. Specified in the same units as the input GeoDataFrame's CRS. Defaults to 1.0.
    target_crs : str or pyproj.CRS, optional
        CRS to reproject the input GeoDataFrame to before regularization.
        If None, no reprojection is performed. Defaults to None.
    simplify : bool, optional
        If True, applies initial simplification to the geometry before
        regularization. Defaults to True.
    simplify_tolerance : float, optional
        Tolerance for the initial simplification step (if `simplify` is True).
        Also used for geometry cleanup steps. Specified in the same units as the input GeoDataFrame's CRS. Defaults to 0.5.
    allow_45_degree : bool, optional
        If True, allows edges to be oriented at 45-degree angles relative
        to the main direction during regularization. Defaults to True.
    diagonal_threshold_reduction : float, optional
        Reduction factor in degrees to reduce the likelihood of diagonal
        edges being created. larger values reduce the likelihood of diagonal edges. Possible values are 0 - 22.5 degrees.
        Defaults to 15 degrees.
    allow_circles : bool, optional
        If True, attempts to detect polygons that are nearly circular and
        replaces them with perfect circles. Defaults to True.
    circle_threshold : float, optional
        Intersection over Union (IoU) threshold used for circle detection
        (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
    num_cores : int, optional
        Number of CPU cores to use for parallel processing. If 1, processing
        is done sequentially. Defaults to 0 (all available cores).
    include_metadata : bool, optional
        If True, includes metadata about the regularization process in the
        output GeoDataFrame. Defaults to False.
    neighbor_alignment : bool, optional
        If True, aligns the polygons with their neighbors after regularization.
        Defaults to False.
    neighbor_search_distance : float, optional
        Search radius used to identify neighboring polygons for alignment (if `align_with_neighbors` is True).
        Specified in the same units as the input GeoDataFrame's CRS. Defaults to 100.0.
    neighbor_max_rotation : float, optional
        Direction threshold for aligning with neighbors (if
        `align_with_neighbors` is True). Defaults to 10 degrees.

    Returns:
    --------
    geopandas.GeoDataFrame
        A new GeoDataFrame with regularized polygon geometries. Original
        attributes are preserved. Geometries that failed processing might be
        dropped.
    """  # noqa: E501, W505
    regularized_geodataframe = buildingregulariser.regularize_geodataframe(
        geodataframe=geodataframe,
        parallel_threshold=parallel_threshold,
        target_crs=target_crs,
        simplify=simplify,
        simplify_tolerance=simplify_tolerance,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        num_cores=num_cores,
        include_metadata=include_metadata,
        neighbor_alignment=neighbor_alignment,
        neighbor_max_rotation=neighbor_max_rotation,
        neighbor_search_distance=neighbor_search_distance,
    )

    return regularized_geodataframe
