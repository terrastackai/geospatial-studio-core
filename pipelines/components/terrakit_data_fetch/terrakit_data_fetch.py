# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
The TerraKit process will query data from a range of different data connectors
"""

# Dependencies
# pip install terrakit==0.1.0 requests opentelemetry-distro opentelemetry-exporter-otlp tenacity

import os
import json
import numpy as np
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
    retry_if_exception_type,
    before_sleep_log,
)
from terrakit import DataConnector
from terrakit.download.geodata_utils import save_data_array_to_file
from terrakit.download.transformations.scale_data_xarray import scale_data_xarray
from terrakit.download.transformations.impute_nans_xarray import impute_nans_xarray
from gfm_data_processing.metrics import MetricManager
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.exceptions import GfmDataProcessingException
from terrakit_cache import TerrakitPVCacheManager

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

# Initialize cache manager using existing /data mount with /cache subfolder
cache_manager = TerrakitPVCacheManager(
    cache_dir=os.getenv("TERRAKIT_CACHE_DIR", "/data/cache"),
    cache_ttl_days=int(os.getenv("TERRAKIT_CACHE_TTL_DAYS", "30")),
    max_cache_size_gb=float(os.getenv("TERRAKIT_CACHE_MAX_SIZE_GB")) if os.getenv("TERRAKIT_CACHE_MAX_SIZE_GB") else None,
    enabled=os.getenv("TERRAKIT_CACHE_ENABLED", "true").lower() == "true"
)


def to_decibels(linear):
    return 10 * np.log10(linear)


def s1grd_to_decibels(da, modality_tag):
    if modality_tag == "S1GRD":
        da[0, 0, :, :] = to_decibels(da[0, 0, :, :])
        da[0, 1, :, :] = to_decibels(da[0, 1, :, :])
    return da


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type((RuntimeError, ConnectionError, OSError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def fetch_data_with_retry(dc, collection_name, data_date, bbox, maxcc, band_names, save_filepath, task_folder):
    """
    Fetch data from connector with automatic retry on network errors.
    Retries up to 3 times with 5 second delays for network-related errors:
    - RuntimeError (includes RasterioIOError)
    - ConnectionError
    - OSError (includes CURL errors)
    """
    logger.info(f"Attempting to fetch data for collection: {collection_name}")
    return dc.connector.get_data(
        data_collection_name=collection_name,
        date_start=data_date,
        date_end=data_date,
        bbox=bbox,
        maxcc=maxcc,
        bands=band_names,
        save_file=save_filepath,
        working_dir=task_folder,
    )


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
            imputed_file_path = f"{task_folder}/{task_id}_{modality_tag}_{output_file_date}_imputed{file_suffix}.tif"
            
            band_names = list(band_dict.get("band_name") for band_dict in model_input_data_spec["bands"])

            # Generate cache key
            cache_key = cache_manager.get_cache_key(
                bbox=bbox,
                date=data_date,
                collection_name=collection_name,
                band_names=band_names,
                maxcc=maxcc,
                modality_tag=modality_tag,
                transform=model_input_data_spec.get("transform")
            )
            
            # Check cache first (in /data/cache)
            cached_data = cache_manager.get_cached_files(cache_key)
            
            if cached_data:
                # Cache hit - copy from /data/cache to task folder
                logger.info(f"🎯 Using cached data for {modality_tag} on {data_date}")
                
                original_pv_path = cached_data["original_pv_path"]
                imputed_pv_path = cached_data["imputed_pv_path"]
                
                # Copy (or hardlink) from cache to task folder
                success_original = cache_manager.copy_cached_file(original_pv_path, save_filepath)
                success_imputed = cache_manager.copy_cached_file(imputed_pv_path, imputed_file_path)
                
                if success_original and success_imputed:
                    original_input_images += [save_filepath]
                    imputed_input_images += [imputed_file_path]
                    logger.info(f"✅ Successfully retrieved cached files")
                    continue  # Skip to next modality
                else:
                    logger.warning(f"⚠️ Failed to copy cached files, fetching from Terrakit...")
            
            # Cache miss or copy failed - fetch from Terrakit
            logger.info(f"🌍 Fetching data from Terrakit for {modality_tag} on {data_date}")
            
            # Use tenacity for automatic retry on network errors
            da = fetch_data_with_retry(
                dc=dc,
                collection_name=collection_name,
                data_date=data_date,
                bbox=bbox,
                maxcc=maxcc,
                band_names=band_names,
                save_filepath=save_filepath,
                task_folder=task_folder,
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
            dai = impute_nans_xarray(dai, nodata_value=nodata_value)
            save_data_array_to_file(dai, imputed_file_path, imputed=True)
            
            original_input_images += [save_filepath]
            imputed_input_images += [imputed_file_path]
            
            # Cache the files to /data/cache
            cache_metadata = {
                "date": data_date,
                "bbox": bbox,
                "collection": collection_name,
                "bands": band_names,
                "maxcc": maxcc,
                "modality": modality_tag,
                "nodata_value": float(nodata_value),
                "transform": model_input_data_spec.get("transform"),
                "inference_id": inference_id,
                "task_id": task_id
            }
            
            cache_manager.cache_files(
                cache_key=cache_key,
                original_file_path=save_filepath,
                imputed_file_path=imputed_file_path,
                metadata=cache_metadata
            )

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
