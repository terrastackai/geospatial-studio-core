# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
The operator to pull and pre-process input data from pre-signed URLs.
"""

# pip install rasterio numpy opentelemetry-distro opentelemetry-exporter-otlp

import copy
import json
import os
import sys
import time

from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.exceptions import GfmDataProcessingException
from gfm_data_processing.metrics import MetricManager
from gfm_data_processing.raster_data_operations import impute_nans, verify_input_image
from preprocessing_helper.user_store_download_operations import (
    check_url_input,
    download_pre_signed_url,
)
from sqlalchemy import create_engine, text

# Uncomment for local testing
# import dotenv
# dotenv.load_dotenv()

######################################################################################################
### Grab the inputs from pipeline environment variables
######################################################################################################

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# get data source index
data_source_index = int(os.environ.get("data_source_index", 0))

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

orchestrate_db_uri = os.getenv("orchestrate_db_uri", "")

db_orchestration = os.environ.get("db_orchestration", "True")

inf_task_table = os.getenv("inference_task_table", "task")

process_id = os.getenv("process_id", "url-connector")

stop_exit_code = int(os.getenv("stop_exit_code", 177))

metric_manager = MetricManager(component_name=process_id)

logger.info("********* Loading and preparing input imagery **********")
fst = time.time()

is_add_layer_task = False
new_output_files = []
output_image_list = []

# helper functions
# detect multi-input for model
def expects_multi_input(task_dict):
    specs = task_dict.get("model_input_data_spec") or []
    return len(specs) > 1
# map downloaded images model input by file_suffix
def map_outputs_to_specs(output_image_list, model_input_data_spec):
    mapped = {}

    for spec in model_input_data_spec:
        suffix = spec.get("file_suffix")
        modality_tag = spec.get("modality_tag")

        if not suffix or not modality_tag:
            continue

        matches = [
            image_dict for image_dict in output_image_list
            if image_dict.get("original_image", "").endswith(suffix)
        ]

        if len(matches) != 1:
            raise GfmDataProcessingException(
                f"Expected exactly one downloaded image for modality={modality_tag}, "
                f"suffix={suffix}; found {len(matches)}."
            )

        mapped[modality_tag] = {
            "original_image": matches[0].get("original_image"),
            "imputed_image": matches[0].get("imputed_image"),
            "file_suffix": suffix,
            "collection": spec.get("collection"),
        }

    return mapped

@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def url_connector_single():
    task_dict = {}
    inference_dict = {}
    task_config_path = None
    t1 = fst
    t2 = fst

    try:
        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Preprocessing started: downloading and preparing data.",
        )

        ######################################################################################################
        ### Parse the inference and task configs
        ######################################################################################################

        logger.info("********* Loading inference and task configuration **********")
        inference_config_path = f"{inference_folder}/{inference_id}_config.json"
        task_folder = f"{inference_folder}/{task_id}"
        task_config_path = f"{task_folder}/{task_id}_config.json"

        with open(inference_config_path, "r") as fp:
            inference_dict = json.load(fp)
            is_add_layer_task = "add-layer-sandbox" in inference_dict.get("model_internal_name", "")
        with open(task_config_path, "r") as fp:
            task_dict = json.load(fp)

        ######################################################################################################
        ### Check the URL and download the data
        ######################################################################################################
        new_output_files = []
        # If multimodal
        if len(task_dict["url"]) > 1 and len(inference_dict["model_input_data_spec"]) > 1:
            # Download multimodal data and save the file names
            logger.info(f"********* Starting data pull multimodal data for task: {task_id} **********")

            for i, url in enumerate(task_dict["url"]):
                # Add date in the task_dict
                if "date" not in task_dict:
                    task_dict["date"] = []
                from datetime import datetime

                task_dict["date"].append(datetime.now().strftime("%Y-%m-%d"))
                # Check the URL and download the data
                original_filename, response = check_url_input(
                    url, task_id, inference_id
                )

                logger.info(
                    f"********* Original filename for task: {task_id} : {original_filename} **********"
                )

                # create the multimodal_file_name
                modality = inference_dict["model_input_data_spec"][i].get(
                    "modality_tag", f"modality{i}"
                )
                file_extension = original_filename.rsplit(".", 1)[-1]
                date_str = task_dict["date"][i] if task_dict.get("date") else ""
                new_filename = f"{task_id}_{modality}_{date_str}.{file_extension}"

                logger.info(
                    f"********* New filename for task: {task_id} : {new_filename} **********"
                )

                downloaded_files = download_pre_signed_url(
                    new_filename, response, task_dict.get("date", ""), f"{task_folder}/"
                )
                new_output_files.extend(downloaded_files)

                logger.info(
                    f"********* Downloaded output_files for task: {task_id} : {new_output_files} **********"
                )

        else:
            logger.info(f"********* Starting data pull for task: {task_id} **********")
            filename, response = check_url_input(task_dict["url"], task_id, inference_id)

            new_output_files = download_pre_signed_url(filename, response, task_dict.get("date", ""), f"{task_folder}/")

        if not new_output_files:
            raise GfmDataProcessingException("No files returned from download_pre_signed_url.")

        t1 = time.time()
        logger.info(f"{task_id}: Time taken to download data = {round(t1 - fst, 1)}s")

        ######################################################################################################
        ### Checks on data and imputing NaNs, this is done for only the tasks with inference as next step
        ######################################################################################################

        for new_output_file in new_output_files:
            imputed_image = None
            if ".tif" in new_output_file:
                verify_status_code, verification_msg = verify_input_image(new_output_file)

            if not is_add_layer_task:
                imputed_image = impute_nans(new_output_file, f"{task_folder}/", "")
            output_image_list.append({"original_image": new_output_file, "imputed_image": imputed_image})

        if len(output_image_list) == 0:
            raise GfmDataProcessingException("No files returned from impute NaNs.")
        t2 = time.time()
        logger.info(f"{task_id}: Time taken to impute NaNs = {round(t2 - t1, 1)}s")

        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Preprocessing completed successfully.",
        )

    except GfmDataProcessingException as gfm_ex:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1013",
            message=f"Preprocessing error: {gfm_ex}",
            verbose=True,
        )
        raise

    except Exception as ex:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="9999",
            message=f"Unhandled error during preprocessing: {ex}",
            verbose=True,
        )
        raise

    ######################################################################################################
    ### Update the task config and clean up
    ######################################################################################################

    finally:
        if not task_config_path:
            logger.info(f"{task_id}: Task config path not initialized; skipping config update.")
            return

        multi_input_task = expects_multi_input(task_dict)
        ######################################################################################################
        ### Update the task config and clean up
        ######################################################################################################
        if len(output_image_list) == 1 or multi_input_task:
            with open(task_config_path, "r") as fp:
                task_dict = json.load(fp)

            if len(output_image_list) == 1:
                image_dict = output_image_list[0]

                if image_dict.get("imputed_image"):
                    task_dict["imputed_input_image"] = image_dict.get("imputed_image")
                elif not is_add_layer_task:
                    raise GfmDataProcessingException(
                        f"Imputed file for file {image_dict.get('original_image')} required for non add layer tasks."
                    )

                task_dict["original_input_image"] = image_dict.get("original_image")

            else:
                task_dict["original_input_images"] = [
                    image_dict.get("original_image") for image_dict in output_image_list
                ]

                task_dict["imputed_input_images"] = [
                    image_dict.get("imputed_image") for image_dict in output_image_list
                    if image_dict.get("imputed_image")
                ]

                if not is_add_layer_task and len(task_dict["imputed_input_images"]) != len(output_image_list):
                    raise GfmDataProcessingException(
                        "Every multi-input image requires an imputed image for non add-layer tasks."
                    )

                task_dict["model_input_images"] = map_outputs_to_specs(
                    output_image_list,
                    task_dict.get("model_input_data_spec", []),
                )

            logger.info(f"********* Updated task dictionary: {json.dumps(task_dict)} **********")

            with open(task_config_path, "w") as fp:
                json.dump(task_dict, fp, indent=4)

        else:
            ######################################################################################################
            ###  Create subtasks for each of the tasks
            ###  Create a subfolder for each sub-task and save the task config file to the folder
            ######################################################################################################
            try:
                if db_orchestration == "True":
                    engine = create_engine(orchestrate_db_uri)
                    if "pipeline-steps" in inference_dict:
                        pipeline_steps = inference_dict["pipeline-steps"]
                    else:
                        raise GfmDataProcessingException(f"Missing pipeline steps for: {inference_id}")

                with open(task_config_path, "r") as fp:
                    task_dict = json.load(fp)

                ps_at_index_0 = next(ps for ps in pipeline_steps if ps.get("step_number") == 0)
                ps_at_index_0["status"] = "FINISHED"

                ps_at_index_1 = next(ps for ps in pipeline_steps if ps.get("step_number") == 1)
                ps_at_index_1["status"] = "READY"

                insert_task_sql = f"""INSERT INTO {inf_task_table}(task_id, status, active, pipeline_steps, inference_id, inference_folder, created_by) VALUES """

                for i, image_dict in enumerate(output_image_list):
                    # append subtask index
                    task_dict_temp = copy.deepcopy(task_dict)
                    task_dict_temp["task_id"] = task_dict_temp["task_id"] + "_" + str(i)

                    # mkdir task folder
                    os.makedirs(f'{inference_folder}/{task_dict_temp["task_id"]}', exist_ok=True)
                    os.makedirs(f"{inference_folder}/completed", exist_ok=True)

                    # Update the task config with paths
                    if image_dict.get("imputed_image"):
                        os.rename(
                            image_dict.get("imputed_image"),
                            image_dict.get("imputed_image").replace(task_dict["task_id"], task_dict_temp["task_id"]),
                        )
                        task_dict_temp["imputed_input_image"] = image_dict.get("imputed_image").replace(
                            task_dict["task_id"], task_dict_temp["task_id"]
                        )
                    elif not is_add_layer_task:
                        raise GfmDataProcessingException(
                            f"Imputed file for file {image_dict.get('original_image')} required for non add layer tasks."
                        )

                    os.rename(
                        image_dict.get("original_image"),
                        image_dict.get("original_image").replace(task_dict["task_id"], task_dict_temp["task_id"]),
                    )
                    task_dict_temp["original_input_image"] = image_dict.get("original_image").replace(
                        task_dict["task_id"], task_dict_temp["task_id"]
                    )

                    # write t into task file
                    with open(
                        f'{inference_folder}/{task_dict_temp["task_id"]}/{task_dict_temp["task_id"]}_config.json',
                        "w",
                        encoding="utf-8",
                    ) as file:
                        json.dump(task_dict_temp, file, ensure_ascii=False, indent=4)

                    if i > 0:
                        insert_task_sql = insert_task_sql + ", "

                    insert_task_sql = (
                        insert_task_sql
                        + f"('{task_dict_temp['task_id']}', 'READY', 'True', '{json.dumps(pipeline_steps)}', '{inference_id}', '{inference_folder}', '{inference_dict['user']}')"
                    )

                insert_task_sql = insert_task_sql + ";"

                if db_orchestration == "True":
                    with engine.connect() as conn:
                        insert_task_sql = text(insert_task_sql)
                        print(insert_task_sql)
                        conn.execute(insert_task_sql)
                        conn.commit()

                notify_gfmaas_ui(
                    event_id=inference_id,
                    task_id=task_id,
                    event_status="Url connector sub-tasks added to queue.",
                )

            except GfmDataProcessingException as gfm_ex:
                report_exception(
                    event_id=inference_id,
                    task_id=task_id,
                    error_code="1013",
                    message=f"Preprocessing error: {gfm_ex}",
                    verbose=True,
                )
                raise

            except Exception as ex:
                report_exception(
                    event_id=inference_id,
                    task_id=task_id,
                    error_code="1044",
                    message=f"Url connector planning failed with: {ex}",
                    event_detail_type="Inf:Task:Failed",
                    verbose=True,  # set to False if you want less detail
                    raise_exception=False,
                )
                raise  # Remove this line if you want to continue after error, else it will stop on error.
            finally:
                logger.info(f"{task_id}: ********* {task_id} Complete **********")

        et = time.time()
        logger.info(
            f"{task_id}: Timing summary — "
            f"Download = {round(t1 - fst, 1)}s | "
            f"Impute = {round(t2 - t1, 1)}s | "
            f"Total = {round(et - fst, 1)}s"
        )

        if len(output_image_list) > 1 and not multi_input_task:
            sys.exit(stop_exit_code)


if __name__ == "__main__":
    url_connector_single()
