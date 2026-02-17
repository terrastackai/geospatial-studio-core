# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import json
import math
import glob
import rasterio
import numpy as np
from gfm_data_processing.exceptions import GfmDataProcessingException

from subprocess import check_output


def check_projection(file):
    """
    Check the projection is correct, if not reproject to EPSG:4326

    Args:
        file (str): path to input file

    Output:
        None
    """
    res = os.popen(f"gdalinfo {file} -proj4 -json").read()
    res_json = json.loads(res)
    # WGS84 is the same as EPSG:4326
    if res_json["stac"]["proj:epsg"] != 4326:
        os.system(f"gdalwarp {file} -t_srs EPSG:4326 -dstnodata -9999 {file}_reprojected.tif")
        os.system(f"mv {file}_reprojected.tif {file} ")


def get_raster_bbox(raster_file):
    """
    It extracts a bbox from a raster file.

    :param raster_file: path to raster file
    :return: bbox (left, bottom, right, up)
    """

    with rasterio.open(raster_file) as src:
        bbox = src.bounds

    return bbox


def get_raster_meta(raster_file):
    """
    It extracts the raster's metadata

    :param raster_file: path to raster file
    :return: raster metadata
    """

    with rasterio.open(raster_file) as src:
        out_meta = src.meta

    return out_meta


def get_raster_crs(raster_file):
    """
    It extracts a raster crs.

    :param raster_file: raster_file: path to raster file
    :return: string containing crs
    """
    meta = get_raster_meta(raster_file)

    crs = str(meta["crs"])

    return crs


def get_raster_resolution(raster_file):
    """
    It extracts a raster resolution. It accounts for changes in rasterio >= 1.0

    :param raster_file: path to raster file
    :return: resolution
    """

    transform = get_raster_meta(raster_file)["transform"]

    if isinstance(transform, rasterio.Affine):
        res = transform[0]

    else:
        res = transform[1]

    return res


def get_raster_data(raster_file, band_index=None, replace_nodata=None):
    """
    It extracts the raster's data from a band

    :param raster_file: path to raster file
    :return: raster band data
    """

    with rasterio.open(raster_file) as src:
        out_data = src.read(band_index) if band_index is not None else src.read()

    if replace_nodata is not None:
        meta = get_raster_meta(raster_file)
        out_data = np.where(out_data == meta["nodata"], replace_nodata, out_data)

    return out_data


def write_raster(img_wrt, filename, metadata, compress=True):
    """
    It writes a raster image to file.

    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    if compress:
        metadata["compress"] = "lzw"

    with rasterio.open(filename, "w", **metadata) as dest:
        if len(img_wrt.shape) > 2:
            for i in range(img_wrt.shape[0]):
                dest.write(img_wrt[i, :, :], i + 1)

        else:
            dest.write(img_wrt, 1)


def change_raster_dtype(file, dtype):
    data = get_raster_data(file).astype(dtype)
    meta = get_raster_meta(file)
    meta["dtype"] = dtype

    write_raster(data, file, meta)


def stack_rasters(raster_list, output_file):
    """

    :param raster_list:
    :param output_file:
    :return:
    """

    with rasterio.open(raster_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count=len(raster_list))

    output_dir = output_file.replace(output_file.split("/")[-1], "")

    if not os.path.isdir(output_dir):
        os.system("mkdir {}".format(output_dir))

    # Read each layer and write it to stack
    with rasterio.open(output_file, "w", **meta) as dst:
        for id, layer in enumerate(raster_list, start=1):
            with rasterio.open(layer) as src1:
                dst.write_band(id, src1.read(1))

    return output_file


def get_resample(name: str) -> str:
    """retrieves code for resampling method
    Args:
        name (:obj:`string`): name of resampling method
    Returns:
        method :obj:`string`: code of resample method
    """

    methods = {
        "first": """
import numpy as np
import pandas as pd
def first(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    df = pd.DataFrame(np.argwhere(~np.isnan(in_ar)), columns=['Tile', 'Row', 'Column'])
    indexes_output = df.groupby(['Row', 'Column']).min().reset_index()[['Tile', 'Row', 'Column']]
    tiles = indexes_output['Tile'].unique()
    for tile in tiles:
        tile_indexes = indexes_output[indexes_output['Tile'].values == tile][['Row', 'Column']]
        out_ar[tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)] = in_ar[tile][tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)]
""",
        "last": """
import numpy as np
import pandas as pd
def last(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    df = pd.DataFrame(np.argwhere(~np.isnan(in_ar)), columns=['Tile', 'Row', 'Column'])
    indexes_output = df.groupby(['Row', 'Column']).max().reset_index()[['Tile', 'Row', 'Column']]
    tiles = indexes_output['Tile'].unique()
    for tile in tiles:
        tile_indexes = indexes_output[indexes_output['Tile'].values == tile][['Row', 'Column']]
        out_ar[tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)] = in_ar[tile][tile_indexes['Row'].min():(tile_indexes['Row'].max()+1), tile_indexes['Column'].min():(tile_indexes['Column'].max()+1)]
""",
        "max": """
import numpy as np
def max(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmax(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "min": """
import numpy as np
def min(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmin(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "median": """
import numpy as np
import pickle
def median(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmedian(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
        "average": """
import numpy as np
import pickle
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    nodata = float(kwargs['nodata'])
    nodata=np.array(nodata, dtype=out_ar.dtype)
    in_ar = np.array(in_ar)
    in_ar = np.where(in_ar == nodata, np.nan, in_ar)
    y = np.nanmean(in_ar, axis=0)
    out_ar[:] = np.where(out_ar == np.nan, nodata, y)
""",
    }

    if name not in methods:
        raise ValueError("ERROR: Unrecognized resampling method (see documentation): '{}'.".format(name))

    return methods[name]


def add_pixel_fn(filename: str, resample_name: str, output_type="Float32", nodata=-9999.0) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = f"""  <VRTRasterBand dataType='{output_type}' band='1' subClass='VRTDerivedRasterBand'>"""
    contents = """
    <PixelFunctionType>{0}</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionArguments {2}/>
    <PixelFunctionCode><![CDATA[{1}]]>
    </PixelFunctionCode>"""

    lines = open(filename, "r").readlines()
    lines[3] = header  # FIX ME: 3 is a hand constant
    lines.insert(4, contents.format(resample_name, get_resample(resample_name), f'nodata="{nodata}"'))
    open(filename, "w").write("".join(lines))


def create_mosaic(
    output_file_prefix: str,
    tile_prefix: str,
    in_dir: str,
    out_dir: str = None,
    method="average",
    delete_tiles=False,
    output_type="Float32",
    nodata=-9999,
) -> str:
    """
    It creates a mosaic of raster files using gdalbuildvrt and gdal_translate. Overlap is taken car of by using the mean.

    :param output_file_prefix: prefix of the files to be tiled. If '' every .tif file is merged together. This is also used to
            save the final result
    :param tile_prefix: prefix of the tile files. If '' every .tif file is merged together
    :param in_dir: input dir (with '/' at the end)
    :param out_dir: output dir (with '/' at the end)
    :param method: gdalbuildvrt method to merge
    :param delete_tiles: whether to delete the tiles after having merged or not
    :param output_type: dtype of output, allowed types are UInt8, UInt16, Int16, UInt32, Int32, Float32, Float64
    :param output_type: no data value to use
    :return: path to output file
    """

    # get all files to merge
    files = glob.glob(f"{in_dir}{output_file_prefix}*{tile_prefix}*tif*")

    if len(files) > 0:
        print(f"Files to merge: {str(len(files))}")

        # get tiles type (it assumes all of the same type)
        tile_meta = get_raster_meta(files[0])
        input_dtype = tile_meta["dtype"]

        # if type different than required one
        if output_type.lower() != input_dtype:
            print(f"Converting tiles dtype from {input_dtype} to output type {output_type}")
            for file in files:
                change_raster_dtype(file, output_type.lower())

        # create file names and vrt dataset
        vrt_tmp_file = f"{in_dir}tmp.vrt"
        os.system(f"gdalbuildvrt -vrtnodata {nodata} {vrt_tmp_file} {in_dir}{output_file_prefix}*{tile_prefix}*.tif*")

        output_file = f"{in_dir}{output_file_prefix}.tif" if output_file_prefix != "" else f"{in_dir}output.tif"

        # add required pixel function
        add_pixel_fn(vrt_tmp_file, method, output_type, nodata)

        # compute mosaic to GeoTiff
        os.environ["GDAL_VRT_ENABLE_PYTHON"] = "Yes"
        os.system('gdal_translate -of GTiff -co "TILED=YES" {} {}'.format(vrt_tmp_file, output_file))
        os.environ["GDAL_VRT_ENABLE_PYTHON"] = "None"

        # delete vrt dataset
        if os.path.isfile(vrt_tmp_file):
            os.remove(vrt_tmp_file)

        if delete_tiles:
            os.system("rm {}{}*{}*.tif*".format(in_dir, output_file_prefix, tile_prefix))

        else:
            # if want to keep tiles check whether need to change dtype back to original one
            if output_type.lower() != input_dtype:
                print(f"Converting tiles dtype back to original type {input_dtype}")
                for file in files:
                    change_raster_dtype(file, input_dtype)

        # move file if needed
        if out_dir is not None:
            output_file = output_file.replace(in_dir, out_dir)

            os.system("mv {}{}.tif {}".format(in_dir, output_file_prefix, out_dir))

    else:
        print("No files to merge")
        output_file = ""

    return output_file


def reproject_raster(
    raster_file_in,
    raster_file_out=None,
    dst_crs="EPSG:4326",
    res=None,
    bbox=None,
    width=None,
    height=None,
    resampling="near",
):
    """
    `it reprojects a raster to a new coordinate system or resolution. Adapted from
    https://rasterio.readthedocs.io/en/latest/topics/reproject.html

    :param raster_file_in: path to input raster file
    :param raster_file_out: path to output raster file
    :param dst_crs: destination crs
    :param res: destination resolution
    :param bbox: bbox of output (left, bottom, right, up)
    :param width: width of output
    :param height: height of output
    :param resampling: resampling method to use (from gdalwarp)
    :return: raster_file_out
    """

    if (raster_file_out is not None) & (raster_file_out != raster_file_in):
        if os.path.isfile(raster_file_out):
            os.system("rm " + raster_file_out)

        raster_file_out_original = raster_file_out

    else:
        raster_file_out_original = None
        raster_file_out = raster_file_in.replace(".tif", "_tmp.tif")

    reproject_string = "gdalwarp"

    if dst_crs is not None:
        reproject_string = f"{reproject_string} -t_srs {dst_crs}"

    if (dst_crs is not None) & (bbox is not None):
        reproject_string = (
            f"{reproject_string} -te_srs {dst_crs} -te {str(bbox[0])} {str(bbox[1])} {str(bbox[2])} {str(bbox[3])}"
        )

    if (width is not None) & (height is not None):
        reproject_string = f"{reproject_string} -ts {str(width)} {str(height)}"

    if res is not None:
        reproject_string = f"{reproject_string} -tr {str(res)}"

    if resampling is not None:
        reproject_string = f"{reproject_string} -r {resampling}"

    if reproject_string != "gdalwarp":
        reproject_string = f"{reproject_string} {raster_file_in} {raster_file_out}"
        print(reproject_string)
        os.system(reproject_string)

        if raster_file_out_original is None:
            os.system(f"rm {raster_file_in}")
            os.system(f"mv {raster_file_out} {raster_file_in}")
            raster_file_out = raster_file_in

    else:
        print("No arguments given to reproject")
        raster_file_out = None

    return raster_file_out


def match_raster_to_target(input_file, target_file, output_suffix="_padded", resampling_method="mode"):
    """
    It matches an input raster to a target raster (e.g. to be used to compare rasters pixel-to-pixel)

    :param input_file: path to input raster file
    :param target_file: path to target input file
    :param output_suffix: suffix to add to file name
    :param resampling_method: as implemented by gdalwarp https://gdal.org/programs/gdalwarp.html
    :return: It returns the path of the output file.
    """

    target_meta = get_raster_meta(target_file)

    res = get_raster_resolution(target_file)
    bbox = get_raster_bbox(target_file)
    crs = get_raster_crs(target_file)
    padded_input_file = input_file.replace(".tif", output_suffix + ".tif")

    padded_input_file = reproject_raster(
        input_file,
        padded_input_file,
        dst_crs=crs,
        bbox=bbox,
        width=target_meta["width"],
        height=target_meta["height"],
        resampling=resampling_method,
    )

    return padded_input_file


def verify_input_image(image) -> tuple[int, str]:
    """
    Verify input dimensions for supplied image

    Args:
        image (str): image file path
        standard_dimensions (int): expected size of image

    Output:
        tuple[int, str]: [verification_status_code, verification_msg]
    """
    res = os.popen(f"gdalinfo {image} -json").read()
    res_json = json.loads(res)

    # Check if image is geotiff
    if res_json["driverShortName"] != "GTiff":
        return 1007, f"Input {image} is not a GeoTiff."
    else:
        check_projection(image)
        return 200, "Valid input"


def calculate_resolution(meter_resolution, lat):
    # Get length of degrees
    lat_rad = math.radians(lat)
    # Calculate the length of one degree in latitude considering the ellipticity of the earth
    lat_degree_length = 111132.954 - 559.822 * math.cos(2 * lat_rad) + 1.175 * math.cos(4 * lat_rad)
    # Calculate the length of one degree in longitude based on the latitude and the earth radius
    lon_degree_length = (math.pi / 180) * math.cos(lat_rad) * 6378137.0
    # Get resolution
    resolution_lat = meter_resolution / lat_degree_length
    resolution_lon = meter_resolution / lon_degree_length

    return resolution_lat, resolution_lon


def pad_bbox(padding_degrees, bbox):
    """
    Add padding to bounding box to help with edge artifacts.

    Args:
        padding_degrees (float): number of degrees to add as border to bbox
        bbox (list(float)): original bounding box [min_lon, min_lat, max_lon, max_lat]

    Output:
        padded_bbox (list(float)): bouning box with border of padding [min_lon, min_lat, max_lon, max_lat]

    """
    return [
        bbox[0] - padding_degrees,
        bbox[1] - padding_degrees,
        bbox[2] + padding_degrees,
        bbox[3] + padding_degrees,
    ]


def impute_nans(file, inputs_folder, event_id):
    """
    Impute nan to improve inference performance

    file (str): path to input tif image
    inputs_folder (str): path to inputs folder
    event_id (str): inference event id

    returns path to imputed image
    """
    bands = []
    # Get meta before changing the data type
    meta = get_raster_meta(file)
    # Convert to float32 for the imputation of nans
    data = change_raster_dtype(file, np.float32)
    data = get_raster_data(file)
    data[data == -9999] = np.nan
    # Replace max float32 number with NaN if nodata value is missing
    data[data == np.finfo(np.float32).max] = np.nan
    # Find median values for each band prior to imputation
    medians = {}
    for i in range(0, meta["count"]):
        median_val = np.nanmedian(data[i])
        # If any of the bands have no valid pixels for an EIS or openEO request then
        # notify user and quit inference. For other requests just notify the user.
        if np.isnan(median_val):
            output_text = "One of input image bands has no valid pixels."
            raise GfmDataProcessingException(output_text)

        medians[i] = median_val
    # impute with nearest neighbor
    for i in range(1, meta["count"] + 1):
        command = f"gdal_fillnodata.py -b {i} {file} {inputs_folder}band{i}.py"
        os.system(command)
        bands.append(f"{inputs_folder}band{i}.py")
    # stack imputed bands and save raster
    stack_rasters(bands, inputs_folder + "imputed.tif")
    data = get_raster_data(inputs_folder + "imputed.tif")
    os.system(f"rm -r {inputs_folder}imputed.tif")
    for band in bands:
        os.system(f"rm -r {band}")
    # Impute remainins nodata values to the median for each band
    for i in range(meta["count"]):
        data[i] = np.where(data[i] == -9999, medians[i], data[i])
    if "int" in meta["dtype"]:
        data = data.astype(int)
    write_raster(data, file.replace(".tif", "_imputed.tif"), meta)

    return file.replace(".tif", "_imputed.tif")


def resize_image(image, input_image=None, bbox=None, output_folder="./"):
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

    if input_image:
        get_bbox_cmd = f"""gdalinfo -json {input_image} | jq '(.cornerCoordinates.upperLeft[0] | tostring) + " " + (.cornerCoordinates.upperLeft[1] | tostring ) + " " + (.cornerCoordinates.lowerRight[0] | tostring) + " " + (.cornerCoordinates.lowerRight[1] | tostring )'"""
        input_image_bbox = check_output([get_bbox_cmd], shell=True, text=True).replace('"', "").replace("\n", "")
        os.system(f"mv {image} {image}-temp.tif")
        command = f"gdal_translate -projwin {input_image_bbox} -of GTiff {image}-temp.tif {image}"
    elif bbox:
        command = f"gdal_translate -projwin {bbox[0]} {bbox[3]} {bbox[2]} {bbox[1]} -of GTiff {image} {output_folder}output.tif"

    os.system(command)
    os.system(f"rm {image}-temp.tif")
