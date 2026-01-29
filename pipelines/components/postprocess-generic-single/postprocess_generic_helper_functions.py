# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import time
import glob
import json
import rasterio
import rioxarray
import numpy as np
import pandas as pd
import xarray as xr
from uuid import uuid4
import geopandas as gpd
from pathlib import Path
from zipfile import ZipFile
from rasterio import features
from typing import List, Union
from subprocess import check_output
from shapely.geometry import Polygon, box

from gfm_data_processing.common import logger
from gfm_data_processing import raster_data_operations as rdo
from postprocess_regularization import (
    raster_to_vector,
    adaptive_regularization,
    regularization,
    hybrid_regularization,
)

# LULC data location
LULC_TILE_ROOT = os.environ.get("LULC_SHARED_DATA_ROOT", "/auxdata/lulc/lc2021/")
LULC_TILE_SHAPEFILE = os.environ.get("LULC_TILE_SHAPEFILE", "/auxdata/lulc/tiles.shp")
LAND_POLYGON_PATH = os.environ.get("LAND_POLYGON_PATH", "/auxdata/general/land_polygons.shp")

# Cloud mask
S2_CLOUD_MASK = [3, 8, 9]  # Cloud shadows, Cloud medium probability, Cloud high probability
HLS_CLOUD_MASK = ["0", "0", "0", "0", "1", "1", "1", "1"]
CLOUD_VALUE = 999

# Snow or ice mask
S2_SNOW_OR_ICE_MASK = [11]  # Snow or ice
HLS_SNOW_OR_ICE_MASK = ["0", "0", "0", "1", "0", "0", "0", "0"]
SNOW_OR_ICE_VALUE = 998

# Permanent water mask
S2_PERMANENT_WATER_MASK = [6]  # Permanent water using S2 SCL
S2_LULC_PERMANENT_WATER_MASK = [1]  # https://www.arcgis.com/home/item.html?id=cfcb7609de5f478eb7666240902d4d3d
S2_LULC_PERMANENT_WATER_BAND_INDEX = 0
HLS_PERMANENT_WATER_MASK = ["0", "0", "1", "0", "0", "0", "0", "0"]
PERMANENT_WATER_VALUE = 997

# Regularization techniques
REGULARIZATION_TECHNIQUES = {
    "adaptive_regularization": adaptive_regularization,
    "regularization": regularization,
    "hybrid_regularization": hybrid_regularization,
}


def resize_image(image, input_image=None, bbox=None):
    """
    Crop image to provided bounding box

    Args:
        image (str): path to input image
        input_image (str): path to the original model input image to match size to
        bbox (list(float)): bounding box to crop to [min_lon, min_lat, max_lon, max_lat]
        output_folder (str): Path to output folder destination

    Output:
        None

    """

    # logger.debug(f"Resizing image: {image}")

    def __resize_image(input_image):
        if input_image:
            get_bbox_cmd = f"""gdalinfo -json {input_image} | jq '(.cornerCoordinates.upperLeft[0] | tostring) + " " + (.cornerCoordinates.upperLeft[1] | tostring ) + " " + (.cornerCoordinates.lowerRight[0] | tostring) + " " + (.cornerCoordinates.lowerRight[1] | tostring )'"""
            input_image_bbox = check_output([get_bbox_cmd], shell=True, text=True).replace('"', "").replace("\n", "")
            os.system(f"cp {image} {image}-temp.tif")
            command = f"gdal_translate -projwin {input_image_bbox} -of GTiff {image}-temp.tif {image}"
        elif bbox:
            command = (
                f"gdal_translate -projwin {bbox[0]} {bbox[3]} {bbox[2]} {bbox[1]} -of GTiff {image}-temp.tif {image}"
            )

        os.system(command)
        os.system(f"rm {image}-temp.tif")

    if isinstance(image, list):
        for i in image:
            __resize_image(i)
    else:
        __resize_image(image)


def read_json_with_retries(filepath, max_retries=5, base_delay=0.5):
    """
    Tries to read a JSON file with retries in case of a JSONDecodeError.
    """
    for attempt in range(1, max_retries + 1):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.info(f"Attempt {attempt}: Failed to decode JSON - {e}")
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))  # Exponential backoff
        except FileNotFoundError as e:
            logger.info(f"Attempt {attempt}: File not found - {e}")
            if attempt == max_retries:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)))


def hls_masking(xds, ids, fmask_index=6, mask_bits=["0", "0", "0", "0", "1", "1", "1", "1"], mask_to_value=999):
    logger.info(f">>>>>>>>>>>> Performing mask for scenes {mask_bits} with masked values set to {mask_to_value}")
    qVals = list(np.unique(ids[fmask_index].astype(int)))
    goodQuality = []
    for v in qVals:
        bit_val = list(format(v, "b").zfill(len(mask_bits)))
        if len([i for i, j in zip(bit_val, mask_bits) if (i == "1") & (j == "1")]) == 0:
            goodQuality.append(v)
    goodMask = np.isin(ids[fmask_index].astype(int), goodQuality, invert=True)
    goodMask = np.expand_dims(goodMask, axis=0)
    goodMask = np.repeat(goodMask, 7, axis=0)
    xdsg = np.ma.MaskedArray(xds, goodMask)
    xds.data = xr.DataArray(xdsg)
    return xds


def s2_masking(xds, ids, fmask_index, scene_classification, mask_to_value):
    """
    S2 mask for a given array of scene classes

    Parameters
    ----------
    xds : xarray
        prediction
    ids : xarray
        original
    fmask_index: int
        band lookup for mask information
    scene_classification: list
        list of scene classes from sentinel-2 scene classification layer to mask
        https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/
    mask_to_value: int
        Mask pixel to this value

    Returns
    -------
    xds
        masked prediction as xarray dataset
    """
    logger.info(
        f">>>>>>>>>>>> Performing mask for scenes {scene_classification} with masked values set to {mask_to_value}"
    )
    mask = ids[fmask_index].values == scene_classification[0]  # initalise the mask array
    for scene_class in range(1, len(scene_classification)):
        mask = np.logical_or(
            mask, ids[fmask_index].values == scene_classification[scene_class]
        )  # append to the mask array for each class in scene_classification array.
    xds = xds.where(np.bitwise_not(mask))
    return xds.fillna(mask_to_value)


def mask_and_set_to(xds, ids, masking_type, inference_dict, i=0):
    # Get masking parameters
    if "data_connector_config" not in inference_dict.keys():
        logger.error("Data connector configuration unavailable. Unable to perform mask")
        return xds
    masking_config = inference_dict["data_connector_config"][i].get(masking_type, None)
    logger.debug(f"Masking config set to: {masking_config}")
    mask_band_index = None
    if masking_config is not None:
        logger.debug(f"Starting {masking_type}...")
        mask_band = masking_config.get("band", None)
        if mask_band is not None:
            encoding_type = masking_config.get("encoding", None)
            if encoding_type is not None and "bands" in inference_dict["model_input_data_spec"][i]:
                model_input_data_spec_bands = inference_dict["model_input_data_spec"][i]["bands"]
                if encoding_type == "sentinel2_lulc":
                    mask_band_index = S2_LULC_PERMANENT_WATER_BAND_INDEX
                else:
                    mask_band_index = search_band_dict(model_input_data_spec_bands, "band_name", f"{mask_band}")
                    if mask_band_index is not None:
                        mask_band_index = mask_band_index.get("index")
            else:
                logger.error(f"No encoding given for {masking_type}")
    if mask_band_index is None:
        logger.exception("No mask band index found. Unable to perform mask")
        return xds
    # Create mask for cloud
    if encoding_type == "sentinel2_scl" or "sentinel2_lulc":
        if masking_type == "cloud_masking":
            mask_values = S2_CLOUD_MASK
            values_to_set = CLOUD_VALUE
        elif masking_type == "snow_ice_masking":
            mask_values = S2_SNOW_OR_ICE_MASK
            values_to_set = SNOW_OR_ICE_VALUE
        elif masking_type == "permanent_water_masking":
            mask_values = S2_PERMANENT_WATER_MASK
            values_to_set = PERMANENT_WATER_VALUE
            if encoding_type == "sentinel2_lulc":
                mask_values = S2_LULC_PERMANENT_WATER_MASK
        else:
            logger.exception(
                f">>>>>>>>>>>> Masking type '{masking_type}' is not valid. Type must be one of 'cloud_masking', 'snow_ice_masking' or 'permanent_water. Update the inference configuration file to resolve. <<<<<<<<<<"
            )  # throw error
            return xds  # TODO: check - do we want to return without masking if this step fails
        # Perform masking for s2
        logger.info(
            f">>>>>>>>>>>> Performing mask for {masking_type} with encoding type: {encoding_type}, mask_band_index: {mask_band_index}, mask_values: {mask_values}"
        )
        try:
            xds = s2_masking(xds, ids, mask_band_index, mask_values, values_to_set)
        except Exception as e:
            logger.exception(f"Postprocessing failed...{e}")
    elif encoding_type == "hls_fmask":
        if masking_type == "cloud_masking":
            mask_values = HLS_CLOUD_MASK
            values_to_set = CLOUD_VALUE
        elif masking_type == "snow_ice_masking":
            mask_values = HLS_SNOW_OR_ICE_MASK
            values_to_set = SNOW_OR_ICE_VALUE
        elif masking_type == "permanent_water_masking":
            mask_values = HLS_PERMANENT_WATER_MASK
            values_to_set = PERMANENT_WATER_VALUE
        else:
            logger.exception(
                f">>>>>>>>>>>> Masking type '{masking_type}' is not valid. Type must be one of 'cloud_masking', 'snow_ice_masking' or 'permanent_water. Update the inference configuration file to resolve. <<<<<<<<<<"
            )  # throw error
            return xds  # TODO: check - do we want to return without masking if this step fails
        # Perform masking for hls
        logger.info(
            f">>>>>>>>>>>> Performing mask for {masking_type} with encoding type: {encoding_type}, mask_band_index: {mask_band_index}, mask_values: {mask_values}"
        )
        try:
            xds = hls_masking(xds, ids, mask_band_index, mask_values, values_to_set)
        except Exception as e:
            logger.exception(f"Postprocessing failed...{e}")
    else:
        logger.info(">>>>>>>>>>>> Valid cloud mask encoding not provided <<<<<<<<<<")
    return xds  # TODO: check - do we want to return without masking if this step fails


# def mask_ocean(input_image_path, output_image_path):
#     land_polygon_path = os.environ['land_polygon_path']
#     xds = rioxarray.open_rasterio(input_image_path)
#     xds.rio.write_nodata(-9999, inplace=True)
#     bbox = xds.rio.bounds()

#     cgdf = gpd.read_file(land_polygon_path, bbox=bbox)
#     polygon = box(*bbox)
#     ccgdf = cgdf.clip(polygon)

#     clipped = xds.rio.clip(ccgdf.geometry.values, ccgdf.crs)
#     clipped.rio.write_nodata(-9999, inplace=True)
#     clipped.rio.to_raster(output_image_path)
#     return output_image_path


def mask_from_url(mask_url: str, xds: object, buffer_size_m: float = 100):
    bbox = xds.rio.bounds()

    cgdf = gpd.read_file(mask_url, bbox=bbox)
    polygon = box(*bbox)

    ccgdf = cgdf.clip(polygon)

    original_crs = ccgdf.crs
    ccgdf.to_crs(epsg=3857, inplace=True)
    ccgdf["geometry"] = ccgdf.buffer(buffer_size_m)
    ccgdf.to_crs(crs=original_crs, inplace=True)

    shapes = ((geom, value) for geom, value in zip(ccgdf.geometry, [90] * len(ccgdf.geometry)))

    features.rasterize(shapes=shapes, fill=0, out=xds, transform=xds.rio.transform())
    return xds


def mask_ocean(xds):
    # xds = rioxarray.open_rasterio(input_image_path)
    # xds.rio.write_nodata(-9999, inplace=True)
    bbox = xds.rio.bounds()

    cgdf = gpd.read_file(LAND_POLYGON_PATH, bbox=bbox)
    polygon = box(*bbox)
    ccgdf = cgdf.clip(polygon)

    xds = xds.rio.clip(ccgdf.geometry.values, ccgdf.crs)
    # clipped.rio.write_nodata(-9999, inplace=True)
    # clipped.rio.to_raster(output_image_path)
    return xds


def make_rgb(model_input_original_image: Union[str, list], inference_dict):
    def _make_rgb_single_file(model_input_original_image: str, i: int = 0):
        xds = rioxarray.open_rasterio(model_input_original_image)
        nodata_value = xds.attrs.get("_FillValue", -9999)
        xds.rio.write_nodata(nodata_value, inplace=True)

        model_input_data_spec_bands = inference_dict["model_input_data_spec"][i]["bands"]
        logger.info(f"model_input_data_spec_bands: {model_input_data_spec_bands}")
        red_band_index = int(search_band_dict(model_input_data_spec_bands, "RGB_band", "R").get("index"))
        green_band_index = int(search_band_dict(model_input_data_spec_bands, "RGB_band", "G").get("index"))
        blue_band_index = int(search_band_dict(model_input_data_spec_bands, "RGB_band", "B").get("index"))
        rgb_indexes = [red_band_index, green_band_index, blue_band_index]

        rgb_xds = xds[rgb_indexes, :, :]
        rgb_filename = model_input_original_image.replace(".tif", "_rgb.tif")
        rgb_xds.rio.to_raster(rgb_filename)
        logger.info(f"RGB Filename created: {rgb_filename}")
        return rgb_filename, i

    if isinstance(model_input_original_image, list):
        for i, image in enumerate(model_input_original_image):
            logger.info(f"Image in make_rgb loop: {image}")
            try:
                return _make_rgb_single_file(image, i)
            except Exception:
                logger.exception(f"Input image tiff {image} doesn't have RGB bands :")

    else:
        return _make_rgb_single_file(model_input_original_image)


def get_tiles_list(bbox):
    ldf = gpd.read_file(LULC_TILE_SHAPEFILE)
    polygon = box(*bbox)
    ldf = ldf.clip(polygon)
    tiles_list = list(ldf["tile"].values)
    return tiles_list


def get_lulc_tile_for_input(input_image_path):
    """
    Find the file path for the LULC tile corresponding to a given input.

    Parameters
    ----------
    input_image_path (str): path to input image

    Returns
    -------
    lulc_file_path (str)
        path to lulc tile for the input image as xarray dataset
    """
    # Setup output directory
    outputs_dir = "/".join(input_image_path.split("/")[:-1])
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    # Read LULC shapefile
    gdf = gpd.read_file(LULC_TILE_SHAPEFILE)
    # Get input image bbox
    bbox = rdo.get_raster_bbox(input_image_path)
    tile_list = get_tiles_list([bbox[0], bbox[1], bbox[2], bbox[3]])
    # Prepare dataframe
    df_tmp = pd.DataFrame({"file": [input_image_path]})
    df_tmp["geometry"] = box(bbox.bottom, bbox.left, bbox.right, bbox.top)
    gdf_bbox = gpd.GeoDataFrame(df_tmp, geometry="geometry", crs="EPSG:4326")
    # Join lulc shapefile dataframe with model input image dataframe
    # to create a dataframe of lulc tile paths, model input image path
    # and model input bbox.
    join = gpd.sjoin(left_df=gdf, right_df=gdf_bbox, how="right", predicate="intersects")
    # Iterate through the list of relavent tiles, then reproject as required.
    for j in range(len(tile_list)):
        tile_j_value = join[join["tile"] == tile_list[j]]["tile"].values[0]
        file_j_value = join[join["tile"] == tile_list[j]]["file"].values[0]
        if tile_j_value == "nan" or not join["tile"].values[j]:
            continue
        tmp_unique_id = str(uuid4())
        # Match LULC tile to model input image
        output_tmp_path = rdo.match_raster_to_target(
            input_file=LULC_TILE_ROOT + tile_j_value,
            target_file=file_j_value,
            output_suffix=f"_{tmp_unique_id}_padded",
        )
        os.system(f"mv {output_tmp_path} {outputs_dir}/lulc_tile{str(j)}.tif")
    lulc_tiles = glob.glob(outputs_dir + "/lulc_tile*")
    if len(lulc_tiles) > 1:
        print("mosaic")
        lulc_file_path = rdo.create_mosaic(
            "lulc",
            "_tile",
            in_dir=lulc_tiles[0].replace(lulc_tiles[0].split("/")[-1], ""),
            delete_tiles=True,
            output_type="Int32",
            method="max",
        )
    elif len(lulc_tiles) == 1:
        print("single file")
        lulc_file_path = outputs_dir + "/lulc.tif"
        os.system(f"mv {lulc_tiles[0]} {lulc_file_path}")
    return lulc_file_path


def search_band_dict(bands_list, search_key, search_value):
    band_dict_match = (band_dict for band_dict in bands_list if band_dict.get(f"{search_key}") == f"{search_value}")
    band_dict = next(band_dict_match, None)
    return band_dict


def zip_inference_data(task_dir):
    """
    Zip data from inference run and upload to COS buckets

    Args:
        outputs_folder (str): path to outputs folder
        inputs_folder (str): path to inputs folder

    Output:
        zip_location (str): path to zip folder

    """

    # zip the folder and move to completed location
    logger.info("Create archive of task assets")
    zip_location = f"{task_dir}/archive.zip"

    directory = Path(task_dir)

    with ZipFile(zip_location, mode="w") as archive:
        for file_path in directory.rglob("*.tif"):
            archive.write(file_path, arcname=file_path.relative_to(directory))


def regularize_by_technique(
    raster_to_regularize,
    technique: str = "adaptive_regularization",
    attribute_name: str = "buildings",
    **kwargs,
) -> Union[gpd.GeoDataFrame, List[Polygon]]:
    """Regularize using a specific technique

    Parameters
    ----------
    raster_to_regularize : str
        can be prediction or masked prediction file_path
    technique : str
        regularization technique to use
    **kwargs: based on technique chosen
        examples:
            for adaptive_regularization use
                simplify_tolerance: float = 0.5,
                area_threshold: float = 0.9,
                preserve_shape: bool = True,
            for regularization use
                angle_tolerance: float = 10,
                simplify_tolerance: float = 0.5,
                orthogonalize: bool = True,
                preserve_topology: bool = True,

    Returns
    -------
    Union[gpd.GeoDataFrame, List[Polygon]
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    rasterized_output_path = raster_to_regularize.replace(".tif", ".geojson")

    # Convert the pred tif to polygons
    rasterized_vector = raster_to_vector(
        raster_path=raster_to_regularize,
        output_path=rasterized_output_path,
        class_values=[1],
        attribute_name=attribute_name,
        min_area=0,
        threshold=0,
        unique_attribute_value=False,
    )
    regularization_technique_to_run = REGULARIZATION_TECHNIQUES.get(technique, adaptive_regularization)

    regularized_gdf = regularization_technique_to_run(building_polygons=rasterized_vector, **kwargs)

    if attribute_name not in regularized_gdf.columns:
        logger.warning(f" Regularized polgon lacks the attribute name {attribute_name}. Adding it back")
        regularized_gdf[attribute_name] = rasterized_vector[attribute_name]

    return regularized_gdf


def geojson_to_tiff(
    prediction_tif_file_path: str,
    geodataframe: gpd.geodataframe.GeoDataFrame,
    output_tif_file_path,
    attribute_name: str = "buildings",
):
    # Tiff or Geopackage or Shape file

    # Load an existing raster file to get its metadata
    raster = rasterio.open(prediction_tif_file_path, mode="r+")
    raster.nodata = None

    # Rasterize using the existing raster's shape and transform
    rasterized_data = features.rasterize(
        ((geom, value) for geom, value in zip(geodataframe.geometry, geodataframe[attribute_name])),
        # [(geom, val) for geom, val in zip(vector.geometry, vector["id"])],
        out_shape=raster.shape,
        transform=raster.transform,
        all_touched=True,
        nodata=raster.nodata,
        dtype=raster.dtypes[0],
        fill=0,
    )

    # Write to GeoTIFF using the existing raster's profile
    profile = raster.profile
    with rasterio.open(output_tif_file_path, "w", **profile) as dst:
        dst.write(rasterized_data, 1)

    with rasterio.open(
        output_tif_file_path,
        "w",
        driver="GTiff",
        crs=raster.crs,
        transform=raster.transform,
        dtype=raster.dtypes[0],
        count=1,
        width=raster.width,
        height=raster.height,
    ) as dst:
        dst.write(rasterized_data, indexes=1)

    return output_tif_file_path


def save_geodataframe(geodf: gpd, output_geojson_path):
    """Save geodataframe to geojson

    Parameters
    ----------
    geodf : gpd
        Geodataframe
    output_geojson_path : str
        Path to save the geojson

    Returns
    -------
    str
        Path to the saved geojson file
    """

    geodf.to_file(output_geojson_path)

    return output_geojson_path
