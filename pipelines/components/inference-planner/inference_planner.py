# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
The operator to plan out the inference pipeline run.
"""

#  pip install terrakit==0.1.0 sentinelhub sqlalchemy pg8000 joblib pandas geopandas shapely opentelemetry-distro opentelemetry-exporter-otlp


import os
import re
import json

import geopandas as gpd
from shapely import box, to_wkt

from sqlalchemy import create_engine, text

from inference_planner_functions import polygon_to_bbox, check_and_crop_bbox, find_dates_bbox

from gfm_data_processing.metrics import MetricManager
from joblib import Parallel, delayed

from sentinelhub import (
    CRS,
    BBox,
    bbox_to_dimensions,
)

from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.exceptions import GfmDataProcessingException

# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

inference_folder = os.environ.get("inference_folder", "")

inference_id = os.environ.get("inference_id", "test-inference-1")

planner_task_id = os.environ.get("task_id", f"{inference_id}-task_planning")

orchestrate_db_uri = os.getenv("orchestrate_db_uri", "")

db_orchestration = os.environ.get("db_orchestration", "True")

inf_task_table = os.getenv("inference_task_table", "task")

bbox_to_tile_threshold_area = int(os.environ.get("bbox_to_tile_threshold_area", 100000000))

data_source_index = int(os.environ.get("data_source_index", 0))

process_id = os.getenv("process_id", "inference-planner")

metric_manager = MetricManager(component_name=process_id)


@metric_manager.count_failures(inference_id=inference_id, task_id=planner_task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=planner_task_id)
def inference_planner():
    try:
        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=planner_task_id,
            event_status="Planning inference tasks ..",
        )

        gdf = gpd.GeoDataFrame(columns=["inference_id", "task_id", "geometry", "date"], geometry=None, crs="EPSG:4326")

        ######################################################################################################
        ###  Parse the inference dictionary
        ######################################################################################################

        inference_config_path = f"{inference_folder}/{inference_id}_config.json"

        logger.info("********* loading inference configuration **********")

        with open(inference_config_path, "r") as fp:
            inference_dict = json.load(fp)

        ######################################################################################################
        ###  For URLs, add them to the task list
        ######################################################################################################

        tasks = []
        task_counter = 0

        if "urls" in inference_dict["spatial_domain"]:
            if len(inference_dict["spatial_domain"]["urls"]) > 0:
                for u in inference_dict["spatial_domain"]["urls"]:
                    single_date_format = r"^\d{4}-\d{2}-\d{2}$"
                    if len(inference_dict["temporal_domain"]) > task_counter and re.match(
                        single_date_format, inference_dict["temporal_domain"][task_counter]
                    ):
                        tasks.append(
                            {
                                "task_id": f"{inference_id}-task_{task_counter}",
                                "data_connector": "url",
                                "url": u,
                                "date": inference_dict["temporal_domain"][task_counter],
                            }
                        )
                    else:
                        tasks.append(
                            {
                                "task_id": f"{inference_id}-task_{task_counter}",
                                "data_connector": "url",
                                "url": u,
                            }
                        )
                    task_counter += 1

        ######################################################################################################
        ###  For tiles, add them to the task list
        ######################################################################################################

        if "tiles" in inference_dict["spatial_domain"]:
            if len(inference_dict["spatial_domain"]["tiles"]) > 0:
                tiles_tasks = inference_dict["spatial_domain"]["tiles"]
            else:
                tiles_tasks = []

        ######################################################################################################
        ###  For polygons, convert to bbox
        ######################################################################################################

        bboxes = []

        if "polygons" in inference_dict["spatial_domain"]:
            if len(inference_dict["spatial_domain"]["polygons"]) > 0:
                for p in inference_dict["spatial_domain"]["polygons"]:
                    bboxes = bboxes + [polygon_to_bbox(p, 0)]

        ######################################################################################################
        ###  For bboxes (including those from polygons), check size -> tile/pad/switch to tiles
        ######################################################################################################
        logger.info("********* tiling bounding boxes if needed **********")

        bbox_tasks = []

        # if there are any bounding boxes provided by the user add them to the polygon-converted ones
        if "bbox" in inference_dict["spatial_domain"]:
            if len(inference_dict["spatial_domain"]["bbox"]) > 0:
                bboxes = bboxes + inference_dict["spatial_domain"]["bbox"]

        # Now check the size of the bounding boxes.
        if bboxes:
            # find the resolution of the highest res input data source
            highest_resolution = min([X["resolution_m"] for X in inference_dict["data_connector_config"]])

        for bb in bboxes:
            bb_dims = bbox_to_dimensions(BBox(bbox=bb, crs=CRS.WGS84), highest_resolution)
            # if the bbox is too big, switch to tile mode and just find a list of tiles
            if (bb_dims[0] * bb_dims[1]) > bbox_to_tile_threshold_area:
                # TODO: add logic here to find the tiles for a bbox
                logger.info(f"Need to find the tiles for bbox: {bb} its too big {bb_dims}")

            # otherwise, just tile and/or pad the bbox
            else:
                aoi_bboxes, aoi_sizes = check_and_crop_bbox(bb, highest_resolution)
                bbox_tasks = bbox_tasks + [list(X.lower_left) + list(X.upper_right) for X in aoi_bboxes]

        ######################################################################################################
        ###  Check data availability for bbox/polygons
        ######################################################################################################

        try:
            if bbox_tasks:
                logger.info("********* Preparing to search for data availability **********")

                sp_temp_domains = [
                    {"bbox": b, "date_string": t} for b in bbox_tasks for t in inference_dict["temporal_domain"]
                ]
                logger.info(f">>>>>> Found {len(sp_temp_domains)} bbox + spatial domain combos")

                ans = Parallel(n_jobs=8, prefer="threads")(
                    delayed(find_dates_bbox)(inference_dict["data_connector_config"], st["bbox"], st["date_string"], maxcc=inference_dict["maxcc"])
                    for st in sp_temp_domains
                )
                ans = [item for sublist in ans for item in sublist]

                logger.debug(f">>>>>> Found {len(ans)} bbox+date combos")

                for bt in ans:
                    tasks.append(
                        {
                            "task_id": f"{inference_id}-task_{task_counter}",
                            "data_connector": inference_dict["model_input_data_spec"][data_source_index]["connector"],
                            "bbox": bt["bbox"],
                            "date": bt["date"],
                        }
                    )
                    gdf.loc[len(gdf)] = [
                        inference_dict["inference_id"],
                        f"{inference_id}-task_{task_counter}",
                        to_wkt(box(*bt["bbox"])),
                        bt["date"],
                    ]
                    task_counter += 1
        except Exception as ex:
            report_exception(
                event_id=inference_id,
                task_id=planner_task_id,
                error_code="1044",
                message=f"Inference planning failed with: {ex}",
                event_detail_type="Inf:Task:Failed",
                verbose=True,  # set to False if you want less detail
                raise_exception=True,
            )

        # Check that url / bbox tasks were created.
        if len(tasks) == 0:
            raise GfmDataProcessingException("No data available; exit inference planning")

        ######################################################################################################
        ###  Check data availability for tiles
        ######################################################################################################

        # TODO: need to figure this out

        ######################################################################################################
        ###  Create a subfolder for each sub-task and save the task config file to the folder
        ######################################################################################################

        if db_orchestration == "True":
            engine = create_engine(orchestrate_db_uri)
            if "pipeline-steps" in inference_dict:
                pipeline_steps = inference_dict["pipeline-steps"]
            else:
                pipeline_steps = [
                    {"status": "READY", "process_id": "sentinelhub-connector", "step_number": 0},
                    {"status": "WAITING", "process_id": "run-inference", "step_number": 1},
                    {"status": "WAITING", "process_id": "postprocess-generic", "step_number": 2},
                    {"status": "WAITING", "process_id": "push-to-geoserver", "step_number": 3},
                ]

        insert_task_sql = f"""INSERT INTO {inf_task_table}(task_id, status, active, pipeline_steps, inference_id, inference_folder, created_by) VALUES """
        gdf["geometry"] = gpd.GeoSeries.from_wkt(gdf["geometry"])
        gdf = gdf.set_geometry("geometry", crs="EPSG:4326")
        gdf.to_file(f"{inference_folder}/{inference_id}_tasks.geojson", driver="GeoJSON")

        for i, t in enumerate(tasks):
            # mkdir task folder
            os.makedirs(f'{inference_folder}/{t["task_id"]}', exist_ok=True)
            os.makedirs(f"{inference_folder}/completed", exist_ok=True)
            # write t into task file
            with open(f'{inference_folder}/{t["task_id"]}/{t["task_id"]}_config.json', "w", encoding="utf-8") as file:
                json.dump(t, file, ensure_ascii=False, indent=4)

            if i > 0:
                insert_task_sql = insert_task_sql + ", "

            insert_task_sql = (
                insert_task_sql
                + f"('{t['task_id']}', 'READY', 'True', '{json.dumps(pipeline_steps)}', '{inference_id}', '{inference_folder}', '{inference_dict['user']}')"
            )

        insert_task_sql = insert_task_sql + ";"

        if db_orchestration == "True":
            with engine.connect() as conn:
                insert_task_sql = text(insert_task_sql)
                logger.info(insert_task_sql)
                conn.execute(insert_task_sql)
                conn.commit()

        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=planner_task_id,
            event_status="Inference tasks added to queue.",
        )

    except Exception as ex:
        report_exception(
            event_id=inference_id,
            task_id=planner_task_id,
            error_code="1044",
            message=f"Inference planning failed with: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,  # set to False if you want less detail
            raise_exception=False,
        )
        raise  # Remove this line if you want to continue after error, else it will stop on error.
    finally:
        logger.info(f"{inference_id}: ********* {planner_task_id} Complete **********")


if __name__ == "__main__":
    inference_planner()
