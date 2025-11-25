# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This component pushes the results to geoserver.
"""

# Dependencies
# pip install rasterio geopandas geoserver-rest==2.10.0 Jinja2 python-dotenv opentelemetry-distro opentelemetry-exporter-otlp

from push_to_geoserver_helper_functions import *

import os
import re
import json
import glob
import random

from geo.Geoserver import Geoserver
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.metrics import MetricManager


# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

# geoserver_url
geoserver_url = os.environ.get("geoserver_url", "")

# geoserver_username
geoserver_username = os.environ.get("geoserver_username", "")

# geoserver_password
geoserver_password = os.environ.get("geoserver_password", "")

geoserver_verify = os.environ.get("geoserver_verify", "")

process_id = os.getenv("process_id", "push-to-geoserver")

metric_manager = MetricManager(component_name=process_id)


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def push_to_geoserver():
    layers_for_visualization = []  # Move to top so always defined
    layers_bboxes = []  # Move to top so always defined
    try:
        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Pushing data to GeoServer...",
        )

        ######################################################################################################
        ###  Parse the inference and task configs from file
        ######################################################################################################

        inference_config_path = f"{inference_folder}/{inference_id}_config.json"
        with open(inference_config_path, "r") as fp:
            inference_dict = json.load(fp)

        task_folder = f"{inference_folder}/{task_id}"

        task_config_path = f"{task_folder}/{task_id}_config.json"
        with open(task_config_path, "r") as fp:
            task_dict = json.load(fp)

        logger.info(task_dict.get("model_output_image"))

        if "geoserver_push" in inference_dict:
            ######################################################################################################
            ###  Push data to geoserver
            ######################################################################################################

            # check if this is an addlayer or inference task
            is_add_layer_task = "add-layer-sandbox" in inference_dict.get("model_internal_name", "")
            if is_add_layer_task:
                # From task_id extract the index of the original task and use it to get the gp for addlayer
                match_index = re.search(r"-task_(\d+)", task_id)
                indexed_gp = None
                if match_index:
                    index = int(match_index.group(1))
                    indexed_gp = inference_dict["geoserver_push"][index]
                if indexed_gp:
                    inference_dict["geoserver_push"] = [indexed_gp]
                else:
                    inference_dict["geoserver_push"] = []

            for gp in inference_dict["geoserver_push"]:
                layer_name = f"{inference_id}-{gp['layer_name']}"
                workspace = gp["workspace"]
                full_layer_name = f"{workspace}:{layer_name}"

                # handle layer name for timeseries plot
                if gp.get("geoserver_style", {}).get("timeseries"):
                    layer_name = f"{layer_name}-gpkg"

                logger.info(f"Saving {layer_name} to geoserver")
                if "filepath_key" in gp:
                    if gp["filepath_key"] != "":
                        retrieved_file_paths = task_dict[gp["filepath_key"]]
                        if type(retrieved_file_paths) == str:
                            retrieved_file_paths = [retrieved_file_paths]
                elif "file_suffix":
                    if gp["file_suffix"] != "":
                        retrieved_file_paths = sorted(glob.glob(f"{inference_folder}/{task_id}/*{gp['file_suffix']}"))

                geo = Geoserver(
                    geoserver_url,
                    geoserver_username,
                    geoserver_password,
                    {"verify": f"{geoserver_verify}"},
                )

                # use the retrieved_file_paths extension to determine the type of file to be pushed to geoserver
                if ".tif" in retrieved_file_paths[0]:
                    logger.debug(f"Pushing geotiff file {retrieved_file_paths[0]} as imagemosaic to geoserver")
                    # push imagemosaic to geoserver
                    css = add_imagemosaic_to_geoserver(geo, workspace, task_folder, layer_name, retrieved_file_paths)

                    # compute raster bounds
                    bounds = compute_raster_bounds(retrieved_file_paths[0])

                    # get style
                    style = create_raster_style(gp, full_layer_name)

                elif ".nc" in retrieved_file_paths[0]:
                    logger.debug(f"Pushing netcdf file {retrieved_file_paths[0]} to geoserver")
                    # push imagemosaic to geoserver
                    css = add_netcdf_to_geoserver(
                        geo, workspace, retrieved_file_paths[0], layer_name, gp["coverage_name"]
                    )

                    # compute raster bounds
                    bounds = compute_raster_bounds(retrieved_file_paths[0])

                    # get style
                    style = create_raster_style(gp, full_layer_name)
                elif ".zip" in retrieved_file_paths[0] or ".gpkg" in retrieved_file_paths[0]:
                    logger.debug(f"Pushing shapefile or geopackage file {retrieved_file_paths[0]} to geoserver")
                    # push imagemosaic to geoserver
                    store_format = (
                        "shp"
                        if ".zip" in retrieved_file_paths[0]
                        else "gpkg"
                        if ".gpkg" in retrieved_file_paths[0]
                        else None
                    )
                    css = add_vector_to_geoserver(geo, workspace, retrieved_file_paths[0], layer_name, store_format)

                    # compute raster bounds
                    bounds = compute_vector_bounds(retrieved_file_paths[0])

                    # get style
                    style = create_vector_style(gp, full_layer_name)
                else:
                    logger.warning(
                        f"Unknown file {retrieved_file_paths[0]} as extension is neither '.tif', '.nc', '.gpkg', nor '.zip'"
                    )

                # merge css style and bounds only if css is available
                if css:
                    logger.debug(f"{style}: generated sld style")
                    css["layer_style_xml"] = style
                    css["display_name"] = gp.get("display_name", "")
                    css["z_index"] = gp.get("z_index", random.randint(10, 100))
                    if "visible_by_default" in gp:
                        css["visible_by_default"] = gp.get("visible_by_default")

                    layers_for_visualization.append(css)

                    if "bbox" in inference_dict["spatial_domain"] and inference_dict["spatial_domain"]["bbox"]:
                        layers_bboxes.append(inference_dict["spatial_domain"]["bbox"][0])
                    else:
                        layers_bboxes.append(bounds)

        if layers_for_visualization and (inference_id and inference_id != "None"):
            notify_gfmaas_ui(
                event_id=inference_id,
                task_id=task_id,
                event_status="Data visualization ready.",
            )
            notify_gfmaas_ui(
                event_id=inference_id,
                event_detail_type="Inf:Task:LayerReady",
                event_processed_result={
                    "predicted_layers": layers_for_visualization,
                    "event_id": inference_id,
                    "bboxes": layers_bboxes,
                    "output_url": "None",
                    "task_id": task_id,
                },
            )
    except Exception as ex:
        logger.error(
            f"{task_id}: Exception during GeoServer push: {ex}"
        )  # Changed for general errors during GeoServer push
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1043",  # place holder for Geoserver error code
            message=f"Geoserver Push failed with: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=False,
            raise_exception=True,
        )
    finally:
        logger.info(f"{task_id} : *********GeoServer Push Complete**********")


if __name__ == "__main__":
    push_to_geoserver()
