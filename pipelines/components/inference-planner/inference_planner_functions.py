# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


from shapely.geometry import shape
import math
import requests
import os
from datetime import datetime, timedelta

from terrakit import DataConnector

from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
)

from gfm_data_processing.common import logger


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


def polygon_to_bbox(polygon, buffer_size):
    bbox = shape(polygon).bounds
    bbox = list(bbox)
    bbox[0] = bbox[0] - buffer_size
    bbox[1] = bbox[1] - buffer_size
    bbox[2] = bbox[2] + buffer_size
    bbox[3] = bbox[3] + buffer_size
    return bbox


def tile_bbox(aoi_size, bbox, resolution, tile_size_x=2200.0, tile_size_y=2200.0):
    ##### Tile bounding box if it is bigger than 2400
    numLon = math.floor(aoi_size[0] / tile_size_x)
    numLat = math.floor(aoi_size[1] / tile_size_y)

    lonStep = (bbox[2] - bbox[0]) * (tile_size_x / aoi_size[0])
    latStep = (bbox[3] - bbox[1]) * (tile_size_y / aoi_size[1])

    lons = [bbox[0] + (lonStep * X) for X in list(range(0, numLon + 1))] + [bbox[2]]
    lats = [bbox[1] + (latStep * X) for X in list(range(0, numLat + 1))] + [bbox[3]]

    bboxes = []
    aoi_bboxes = []
    aoi_sizes = []

    for x in range(0, numLon + 1):
        for y in range(0, numLat + 1):
            # bboxes = bboxes + [[lons[x], lats[y], lons[x+1], lats[y+1]]]
            aoi_bbox = BBox(bbox=[lons[x], lats[y], lons[x + 1], lats[y + 1]], crs=CRS.WGS84)
            aoi_bboxes = aoi_bboxes + [aoi_bbox]
            aoi_sizes = aoi_sizes + [bbox_to_dimensions(aoi_bbox, resolution=resolution)]

    return aoi_bboxes, aoi_sizes


def check_and_crop_bbox(bbox, resolution):
    # logger.info(f'#~#~#~#~#~#~#~#~# input bounding box = {bbox}')
    # Check expected pixel size (Sentinel Hub is limited to 2500 pixel)
    aoi_bbox = [BBox(bbox=bbox, crs=CRS.WGS84)]
    # logger.info(f'#~#~#~#~#~#~#~#~# aoi bounding box = {aoi_bbox}')
    aoi_size = [bbox_to_dimensions(aoi_bbox[0], resolution=resolution)]
    if any(s > 2400 for s in aoi_size[0]):
        aoi_bbox, aoi_size = tile_bbox(aoi_size[0], bbox, resolution)

    # print(aoi_bbox)
    # print(aoi_size)

    for i, b in enumerate(aoi_size):
        # print(i)
        if any(s < 244 for s in b):
            print(f"Dimension less than 244, will pad - {aoi_size[i]}")
            center_lon = aoi_bbox[i].middle[0]
            center_lat = aoi_bbox[i].middle[1]
            resolution_lat, resolution_lon = calculate_resolution(meter_resolution=resolution, lat=center_lat)
            # Add 1 for buffer
            padding = int(224 / 2) + 50

            new_bbox = list(aoi_bbox[i])

            if aoi_size[i][0] < 224:
                # Add padding to the image
                # logger.debug(f"{event_id}: Increasing longitude bounds")
                new_bbox[0] = center_lon - padding * resolution_lon
                new_bbox[2] = center_lon + padding * resolution_lon
            if aoi_size[i][1] < 224:
                # logger.debug(f"{event_id}: Increasing latitude bounds")
                new_bbox[1] = center_lat - padding * resolution_lat
                new_bbox[3] = center_lat + padding * resolution_lat
            aoi_bbox[i] = BBox(bbox=new_bbox, crs=CRS.WGS84)
            aoi_size[i] = bbox_to_dimensions(aoi_bbox[i], resolution=resolution)
            print(f"New dimensions are {aoi_size[i]}")

    return aoi_bbox, aoi_size


def find_data_bbox(data_connector: str, data_collection: str, bbox: list, date_string: str, maxcc: float = None):

    dc = DataConnector(connector_type=data_connector)
    print(dc.connector.list_collections())

    if "_" in date_string:
         date_start = date_string.split('_')[0]
         date_end = date_string.split('_')[1]
    else:
        date_start = date_string
        date_end = date_string

    print(data_collection)
    print(date_start, date_end)

    unique_dates, results = dc.connector.find_data(data_collection_name=data_collection,
                            date_start=date_start,
                            date_end=date_end,
                            bbox=bbox,
                            maxcc=maxcc)
    
    print(unique_dates)

    return unique_dates


def find_dates_bbox(
    data_connector_config: dict,
    bbox: list,
    date_string: str,
    pre_days: int = 1,
    post_days: int = 1,
    maxcc: float = None,
):
    data_connector_config_first = data_connector_config[0]
    primary_dates = sorted(
        find_data_bbox(
            data_connector=data_connector_config_first["connector"],
            data_collection=data_connector_config_first["collection_name"],
            bbox=bbox,
            date_string=date_string,
            maxcc=maxcc
        )
    )

    if len(data_connector_config) > 1:
        other_dates = []
        for i in range(1, len(data_connector_config)):
            data_connector_config_selected = data_connector_config[i]
            next_collection_data = [
                sorted(
                    find_data_bbox(
                        data_connector_config_selected["connector"],
                        data_connector_config_selected["collection_name"],
                        bbox,
                        date_string,
                        maxcc
                    )
                )
            ]
            if not next_collection_data[0]:
                raise Exception(
                    f"One modality {data_connector_config_selected['collection_name']} does not have any data"
                )
            other_dates = other_dates + next_collection_data

        primary_dates = [datetime.strptime(X, "%Y-%m-%d") for X in primary_dates]
        other_dates = [[datetime.strptime(X, "%Y-%m-%d") for X in Y] for Y in other_dates]

        output_dict = []

        for p in primary_dates:
            od = other_dates[0]

            pre_date = p - timedelta(days=pre_days)
            post_date = p + timedelta(days=post_days)

            time_diffs_tested = [(t - p) if (t >= pre_date) & (t <= post_date) else timedelta(days=100) for t in od]
            time_diffs_test = [(t >= pre_date) & (t <= post_date) for t in od]
            time_diffs_abs = [abs(X) for X in time_diffs_tested]

            closest_index = time_diffs_abs.index(min(time_diffs_abs))
            # print(f"Primary date: {p}     Closest secondary date: {od[closest_index]}   Time difference: {abs(p-od[closest_index])}   Pass/fail: {time_diffs_test[closest_index]} ")

            if time_diffs_test[closest_index] == True:
                output_dict = output_dict + [
                    {"bbox": bbox, "date": [p.strftime("%Y-%m-%d"), od[closest_index].strftime("%Y-%m-%d")]}
                ]

    else:
        output_dict = [{"bbox": bbox, "date": X} for X in primary_dates]

    return output_dict
