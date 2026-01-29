# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
The TerraKit process will query data from a range of different data connectors
"""

# Dependencies
# pip install terrakit==0.1.0 requests opentelemetry-distro opentelemetry-exporter-otlp

import os
import json
import numpy as np
from terrakit import DataConnector
from terrakit.download.geodata_utils import save_data_array_to_file
from terrakit.download.transformations.scale_data_xarray import scale_data_xarray
from terrakit.download.transformations.impute_nans_xarray import impute_nans_xarray
from gfm_data_processing.metrics import MetricManager
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.exceptions import GfmDataProcessingException

# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

process_id = os.getenv("process_id", "terrakit-data-fetch")

metric_manager = MetricManager(component_name=process_id)


def to_decibels(linear):
    return 10 * np.log10(linear)


def s1grd_to_decibels(da, modality_tag):
    if modality_tag == "S1GRD":
        da[0, 0, :, :] = to_decibels(da[0, 0, :, :])
        da[0, 1, :, :] = to_decibels(da[0, 1, :, :])
    return da


@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def terrakit_data_fetch():
    try:
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

        ######################################################################################################
        ###  Add your processing code here
        ######################################################################################################

        logger.info(f"********* starting query: {task_id} **********")

        bbox = task_dict["bbox"]
        maxcc = inference_dict["maxcc"]

        no_of_modalities = len(inference_dict["data_connector_config"])

        imputed_input_images = []
        original_input_images = []

        for i in range(no_of_modalities):
            data_connector_config = inference_dict["data_connector_config"][i]
            model_input_data_spec = inference_dict["model_input_data_spec"][i]
            collection_name = data_connector_config["collection_name"]
            dc = DataConnector(connector_type=model_input_data_spec["connector"])
            logger.info(dc.connector.list_collections())

            if no_of_modalities == 1:
                data_date = task_dict["date"]
                primary_date = data_date
            elif task_dict["date"][i] and no_of_modalities > 1 and task_dict["date"][i] != "":
                data_date = task_dict["date"][i]
                primary_date = task_dict["date"][0]

            notify_gfmaas_ui(
                event_id=inference_id,
                task_id=task_id,
                event_status=f"Querying modality {i+1} of {no_of_modalities}...",
            )

            if "modality_tag" in data_connector_config:
                modality_tag = data_connector_config["modality_tag"]
            else:
                modality_tag = data_connector_config["collection_name"]

            if "file_suffix" in model_input_data_spec:
                file_suffix = "_" + model_input_data_spec["file_suffix"]
            else:
                file_suffix = ""

            if "align_dates" in model_input_data_spec:
                if model_input_data_spec["align_dates"] in ["True", "true"]:
                    output_file_date = primary_date
                else:
                    output_file_date = data_date
            else:
                output_file_date = data_date

            save_filepath = f"{task_folder}/{task_id}_{modality_tag}_{output_file_date}{file_suffix}.tif"
            original_input_images += [save_filepath]

            band_names = list(band_dict.get("band_name") for band_dict in model_input_data_spec["bands"])

            da = dc.connector.get_data(
                data_collection_name=collection_name,
                date_start=data_date,
                date_end=data_date,
                bbox=bbox,
                maxcc=maxcc,
                bands=band_names,
                save_file=save_filepath,
                working_dir=task_folder,
            )
            logger.debug("\n\nRetrieved data cube\n\n")
            logger.debug(da)
            nodata_value = da.attrs.get("_FillValue", -9999)

            if (da.values == 0).all():
                raise GfmDataProcessingException("All band values are zero, data cube retrieved is empty")

            # Convert s1grd from linear to decibels
            if model_input_data_spec.get("transform") == "to_decibels":
                da = s1grd_to_decibels(da, modality_tag=modality_tag)

            # Get scaling factor list from bands list
            model_input_data_spec_scaling_factors = list(
                float(band_dict.get("scaling_factor", 1)) for band_dict in model_input_data_spec["bands"]
            )
            dai = scale_data_xarray(da, model_input_data_spec_scaling_factors)

            # Imputing nans if any are found in data
            imputed_file_path = f"{task_folder}/{task_id}_{modality_tag}_{output_file_date}_imputed{file_suffix}.tif"
            dai = impute_nans_xarray(dai, nodata_value=nodata_value)
            save_data_array_to_file(dai, imputed_file_path, imputed=True)
            imputed_input_images += [imputed_file_path]

        ######################################################################################################
        ###  (optional) if you want to pass on information to later stages of the pipelines,
        ###             add information to the task config file which will be read later
        ######################################################################################################

        with open(task_config_path, "r") as fp:
            task_dict = json.load(fp)

        task_dict["imputed_input_image"] = imputed_input_images
        task_dict["original_input_image"] = original_input_images

        with open(task_config_path, "w") as fp:
            json.dump(task_dict, fp, indent=4)
    except Exception as ex:
        logger.error(f"{inference_id}: Exception {type(ex).__name__}: {ex}", stack_info=True, exc_info=True)
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1040",  # place holder for error code
            message=f"Terrakit connector failed with: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,
            raise_exception=False,
        )
        raise ex
    finally:
        logger.info(f"{inference_id}: *********Terrakit Connector Complete**********")


if __name__ == "__main__":
    terrakit_data_fetch()
