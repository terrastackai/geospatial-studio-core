# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
The operator to pull and pre-process input data from pre-signed URLs.
"""

# pip install rasterio numpy opentelemetry-distro opentelemetry-exporter-otlp

import os
import sys
import time
import json
import copy

from sqlalchemy import create_engine, text
from gfm_data_processing.metrics import MetricManager
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.exceptions import GfmDataProcessingException
from gfm_data_processing.raster_data_operations import impute_nans, verify_input_image
from preprocessing_helper.user_store_download_operations import check_url_input, download_pre_signed_url

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

stop_exit_code = int(os.getenv("stop_exit_code", 9876))

metric_manager = MetricManager(component_name=process_id)

logger.info("********* Loading and preparing input imagery **********")
fst = time.time()

is_add_layer_task = False
new_output_files = []
output_image_list = []


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def url_connector_single():
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
        if len(output_image_list) == 1:
            ######################################################################################################
            ### Update the task config and clean up
            ######################################################################################################
            try:
                with open(task_config_path, "r") as fp:
                    task_dict = json.load(fp)
                if output_image_list[0].get("imputed_image"):
                    task_dict["imputed_input_image"] = output_image_list[0].get("imputed_image")
                elif not is_add_layer_task:
                    raise GfmDataProcessingException(
                        f"Imputed file for file {output_image_list[0].get('original_image')} required for non add layer tasks."
                    )
                task_dict["original_input_image"] = output_image_list[0].get("original_image")

                logger.info(f"********* Updated task dictionary: {json.dumps(task_dict)} **********")

                with open(task_config_path, "w") as fp:
                    json.dump(task_dict, fp, indent=4)

            except GfmDataProcessingException as gfm_ex:
                report_exception(
                    event_id=inference_id,
                    task_id=task_id,
                    error_code="1013",
                    message=f"Preprocessing error: {gfm_ex}",
                    verbose=True,
                )
                raise

            except Exception as update_err:
                logger.warning(f"{task_id}: Failed to update task config with output paths: {update_err}")
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

        if len(output_image_list) > 1:
            sys.exit(stop_exit_code)


if __name__ == "__main__":
    url_connector_single()
