# Standard
import ast
import copy
import os
import re
import threading
import time
from urllib.parse import urlparse

import alog
import numpy as np
import rasterio
from caikit.core import ModuleBase, ModuleConfig, TaskBase, module, task
from caikit.core.exceptions import error_handler
from caikit.core.modules import ModuleSaver
from caikit.core.toolkit.concurrency.destroyable_thread import DestroyableThread
from cerberus import Validator

# terratorch functions
from terratorch.cli_tools import LightningInferenceModel


# Local
from geospatial_extension.blocks.geospatial.convert_checkpoints import convert_old_checkpoints
from geospatial_extension.blocks.geospatial.data_loader import (
    save_to_output_url,
    set_up_folders,
    url_request,
    zip_inference_data,
)
from geospatial_extension.blocks.geospatial.metrics_utils import emit_prometheus_metrics
from geospatial_extension.blocks.geospatial.settings import (
    APPLICATION_DATA_DIR,
    ALLOWED_STORAGE_PATTERNS,
)
from geospatial_extension.blocks.geospatial.utils import (
    cleanup_complete_inference_files,
    report_exception,
    save_inference_output,
    timeit,
    validate_presigned_urls,
    update_grep_config_in_file
)
from geospatial_extension.data_model.image import ImageResult
from webhooks import webhooks
from .metrics import (
    DURATION_SUMMARY_LABELS,
    DURATION_SUMMARY_METRIC,
    EXCEPTIONS_COUNTER,
    EXCEPTIONS_COUNTER_LABELS,
    RUNNING_GAUGE_LABELS,
    RUNNING_GAUGE_METRIC,
)

logger = alog.use_channel("<BRT_TKN_EMBD>")
error = error_handler.get(logger)
os.environ["PROJ_LIB"] = "/opt/miniconda/share/proj"

# TODO: Hacky way to use a semaphore to ensure locks on inference runs.
# There is race for the existing gpu and before we can find a viable solution
# This is being used to ensure only one inference is run at any given time.
# By default concurrent inference runs will wait indefinitely for a semaphore to become available.
semaphore = threading.Semaphore(1)  # Only allow one request at a time

INFERENCE_SCHEMA = {
    "inputs": {
        "type": "list",
        "required": False,
        "schema": {
            "type": "string",
            "empty": False,
            "is_valid_storage_url": True,
        },
    },
    "output": {"type": "string", "required": False, "is_valid_storage_url": True},
    "event_id": {"type": "string", "required": True},
    "model_id": {"type": "string", "required": True},
    "run_async_inference": {"type": "string", "required": False},
    "use_shared_storage": {"type": "string", "required": False, "default": "false"},
}

ASYNC_INFERENCE_SCHEMA = {
    "inputs": {
        "type": "list",
        "required": True,
        "schema": {"type": "string", "empty": False, "is_valid_storage_url": True},
    },
    "output": {
        "type": "string",
        "required": False,
        "is_valid_storage_url": True,
    },
    "event_id": {"type": "string", "required": True},
    "model_id": {"type": "string", "required": True},
    "run_async_inference": {"type": "string", "required": True},
    "use_shared_storage": {"type": "string", "required": False, "default": "false"},
}


class CustomValidator(Validator):
    def _validate_is_valid_storage_url(self, is_valid_storage_url, field, value):
        """Validate that the value is a URL matching one of the storage patterns."""
        if self.root_document.get("use_shared_storage") in [True, "true", "True"]:
            # Skip input urls validation if use_shared_storage is True.
            # This is used where the input images are in a shared volume).
            # The input is a list of image paths (e.g. /opt/input/image1.tif) and the output is a presigned URL.
            return

        if is_valid_storage_url:
            if not self._is_valid_storage_url(value):
                self._error(field, f"URL - {value} is not a supported storage URL.")

    def _is_valid_storage_url(self, url):
        """Check if the URL matches any of the defined storage URL patterns."""
        return any(re.match(pattern, url) for pattern in ALLOWED_STORAGE_PATTERNS)


def open_tiff(fname):

    with rasterio.open(fname, "r") as src:

        data = src.read()

    return data


def write_tiff(img_wrt, filename, metadata):
    """
    It writes a raster image to file.

    :param img_wrt: numpy array containing the data (can be 2D for single band or 3D for multiple bands)
    :param filename: file path to the output file
    :param metadata: metadata to use to write the raster to disk
    :return:
    """

    with rasterio.open(filename, "w", **metadata) as dest:

        if len(img_wrt.shape) == 2:

            img_wrt = img_wrt[None]

        for i in range(img_wrt.shape[0]):
            dest.write(img_wrt[i, :, :], i + 1)

    return filename


def get_meta(fname):

    with rasterio.open(fname, "r") as src:

        meta = src.meta

    return meta


def check_input_output_urls(input_urls: list[str], output_url: str) -> dict:
    url_validation_error = {}
    for url in input_urls:
        is_valid_input_url, error = validate_presigned_urls(url)
        if not is_valid_input_url:
            url_validation_error["inputs"] = {}
            url_validation_error["inputs"][url] = error

    is_valid_output_url, error = validate_presigned_urls(output_url, method="put")
    if not is_valid_output_url:
        url_validation_error["output"] = {}
        url_validation_error["output"][output_url] = error

    return url_validation_error


@task(
    required_parameters={"text": str},
    output_type=ImageResult,
)
class Geospatial(TaskBase):
    pass


@module(
    id="089e235a-ae91-4c67-8d0c-3f0fa1c0e06e",
    name="Geo block",
    version="0.0.1",
    task=Geospatial,
)
class ImageryModule(ModuleBase):
    def __init__(self, model) -> None:
        """
        This function gets called by `.load` and `.train` function
        which initializes this module.
        Args:
            model: model defined in load()
        """
        super().__init__()
        self.model = model
        self.run_async_inference = os.getenv("RUN_ASYNC_INFERENCE", False)

    @emit_prometheus_metrics(
        [
            (
                lambda labels: RUNNING_GAUGE_METRIC.labels(
                    **{**labels, "step": "inference"}
                ).track_inprogress(),
                RUNNING_GAUGE_LABELS,
            ),
            (
                lambda labels: DURATION_SUMMARY_METRIC.labels(
                    **{**labels, "step": "inference"}
                ).time(),
                DURATION_SUMMARY_LABELS,
            ),
        ]
    )
    @timeit(description="Inferencing step", custom_logger=logger)
    def inference(self, event_id, inference_input, outputs_folder):
        """
        Perform model inference
        Args:
            inference_input (str): path to input tif image
            outputs_folder (str): path to output folder location
        Output:
            output_file (str): path to prediction tif image
        """
        # get bbox size from the input image and have it as a metric
        # BBOX_SIZE.set(gdu.get_raster_bbox(inference_output))

        logger.debug(f"{event_id} - Inference Input passed to model: {inference_input}")
        with semaphore:
            logger.debug(f"*********INFERENCE STARTED [{event_id}]**********")
            inference_output = self.model.inference(inference_input)
            inference_output = inference_output.clone().detach().cpu().numpy()

        # Need image metadata for output
        # mask out nodata
        self.meta = get_meta(inference_input)
        mask = open_tiff(inference_input)
        mask = np.where(mask == self.meta["nodata"], 1, 0)

        mask = np.max(mask, axis=0)[None]

        inference_output = np.where(mask == 1, -1, inference_output)

        self.meta["count"] = 1
        self.meta["dtype"] = "int16"
        self.meta["compress"] = "lzw"
        self.meta["nodata"] = -1
        output_file = write_tiff(
            inference_output,
            outputs_folder
            + inference_input.split("/")[-1]
            .replace("_imputed.tiff", "")
            .replace("_imputed.tif", "")
            + "_pred.tif",
            self.meta,
        )
        logger.debug(f"*********INFERENCE ENDED [{event_id}]**********")
        return output_file

    def _run_process_images_async(self, *args, **kwargs):
        kwargs = kwargs.get("runnable_kwargs") or kwargs
        try:
            self.process_images(*args, **kwargs)
        except Exception as exc:
            logger.exception(f"{kwargs['event_id']} - Error when running inference.")
            inference_output = copy.deepcopy(kwargs["input_data"])
            inference_output.update(
                {
                    "model_id": self.model_id,
                    "status": "ERROR",
                    "message": str(exc),
                }
            )
            inference_output_file = os.path.join(
                kwargs["outputs_folder"], "response.json"
            )
            save_inference_output(
                output_data=inference_output,
                file_path=inference_output_file,
                notify=True,
                event_id=kwargs["event_id"],
            )
            cleanup_complete_inference_files(event_id=kwargs["event_id"])
            raise

    def process_images(
        self,
        event_id: str,
        images: dict,
        image_urls: list,
        inputs_folder: str,
        outputs_folder: str,
        input_data: dict,
        details: dict = None,
    ) -> list:
        logger.debug(f"{event_id} - Processing images started.")
        input_data = copy.deepcopy(input_data)

        # Download images if a presigned Url is provided
        if image_urls:
            try:
                images = url_request(
                    urls=image_urls,
                    inputs_folder=inputs_folder,
                    extra_details=details,
                )
                logger.debug(
                    f"{event_id} - Successfully downloaded {len(images)} images from input URLs"
                )
            except Exception:
                logger.exception(f"{event_id} - Error when resolving input urls.")
                raise

        total_images = len(images)
        output_files = []
        for i, image in enumerate(images):
            st = time.time()
            logger.debug(f"Processing Image {i+1} of {total_images}")
            webhooks.notify_gfmaas_ui(
                event_id,
                f"Inference running for image {i+1} out of {total_images}",
                event_details=details,
            )
            # Handle case where there is a transport error
            try:
                output_file = self.inference(event_id, image, outputs_folder)
                output_files.append(output_file)
            except Exception as exception:
                # Check to see if output folder has crashed
                pvc_check = os.system(f"ls {APPLICATION_DATA_DIR}")

                # ls returns 512 if the directory cannot be accessed
                if pvc_check == 512:
                    report_exception(
                        exception,
                        event_id,
                        1031,
                        "S3FS error with attached COS bucket.",
                        extra_details=details,
                    )
                # if no 512 error then error is general inference error
                else:
                    report_exception(
                        exception,
                        event_id,
                        1030,
                        "Inference failed.",
                        extra_details=details,
                    )
                raise

            webhooks.notify_gfmaas_ui(
                event_id,
                f"Inference completed for image {i+1} out of {total_images}",
                event_details=details,
            )

            et = time.time()
            time_taken = str(np.round(et - st, 1))
            logger.debug(f"Time taken is for inference is {time_taken} seconds.")

        logger.debug(f"{event_id} - Processing images completed.")
        inference_output_file = os.path.join(outputs_folder, "response.json")

        output_url = input_data.get("output")
        if output_url:
            logger.debug(
                f"{event_id} - Attempting to upload data zip file to presigned URL."
            )
            # Zip inference data to internal COS bucket
            zip_location = zip_inference_data(
                outputs_folder, inputs_folder, details=details
            )
            # Save inference data to provided output_url or generate pre-signed url for internal COS bucket
            output_url = save_to_output_url(
                zip_location=zip_location,
                output_url=output_url,
                inputs_folder=inputs_folder,
                details=details,
            )

        inference_output = {
            "event_id": event_id,
            "model_id": self.model_id,
            "inputs": input_data.get("inputs", []),
            "output": output_url or output_files,
            "status": "COMPLETED",
        }

        save_inference_output(
            output_data=inference_output,
            file_path=inference_output_file,
            notify=True,
            event_id=event_id,
        )
        cleanup_complete_inference_files(event_id=event_id)
        return inference_output

    @emit_prometheus_metrics(
        [
            (
                lambda labels: RUNNING_GAUGE_METRIC.labels(
                    **{**labels, "step": "all_inference_tasks"}
                ).track_inprogress(),
                RUNNING_GAUGE_LABELS,
            ),
            (
                lambda labels: DURATION_SUMMARY_METRIC.labels(
                    **{**labels, "step": "all_inference_tasks"}
                ).time(),
                DURATION_SUMMARY_LABELS,
            ),
            (
                lambda labels: EXCEPTIONS_COUNTER.labels(**labels).count_exceptions(),
                EXCEPTIONS_COUNTER_LABELS,
            ),
        ]
    )
    def run(self, text: str) -> ImageResult:
        """Run inference.

        Args:
            text (str(dict)): request information including location of pre-processed data and other inference settings.
        Returns:
            ImageResult containing output text
        """
        # Convert string input to dict
        input_data = ast.literal_eval(text)
        event_id = input_data.get("event_id")
        logger.debug("Inference Request Input: %s", input_data)

        async_inference = str(input_data.get("run_async_inference", False))
        shared_storage = str(input_data.get("use_shared_storage", False))
        bool_map = {"true": True, "false": False}
        self.run_async_inference = bool_map.get(async_inference.lower(), False)
        self.use_shared_storage = bool_map.get(shared_storage.lower(), False)

        url_errors = None
        if self.run_async_inference:
            validator = CustomValidator(ASYNC_INFERENCE_SCHEMA, allow_unknown=True)
        else:
            validator = CustomValidator(INFERENCE_SCHEMA, allow_unknown=True)

        if not validator.validate(input_data):
            return ImageResult(
                text=str(
                    {
                        "error": f"Validation Error: {validator.errors}",
                        "status": "FAILED",
                    }
                )
            )

        # TODO: Remove this check to validate urls for both async and sync requests.
        # Has to be placed after the cerebus validator.validate check.
        if self.run_async_inference and not self.use_shared_storage:
            url_errors = check_input_output_urls(
                input_urls=input_data.get("inputs"),
                output_url=input_data.get("output"),
            )

        if url_errors:
            # Validations currently only active on async.
            if self.run_async_inference:
                return ImageResult(
                    text=str(
                        {
                            "error": f"Validation Error",
                            "detail": url_errors,
                            "status": "FAILED",
                        }
                    )
                )

        try:
            self.model_id = self.model.model_id
        except Exception:
            self.model_id = input_data.get("model_id")

        user = input_data.get("user")
        if not user:
            logger.warning(
                f"{event_id} - No user detail found in inference request payload."
            )

        image_urls = None
        images = None
        if "inputs" in input_data:
            if self.use_shared_storage:
                images = input_data["inputs"]
                inputs_folder = images[0].rsplit("/", 1)[0] + "/"
                outputs_folder = inputs_folder.replace("inputs", "outputs")
            else:
                image_urls = input_data.get("inputs")
                inputs_folder, outputs_folder = set_up_folders(event_id)
        elif "predicted_layers" in input_data:
            images = input_data["predicted_layers"]
            inputs_folder = images[0].rsplit("/", 1)[0] + "/"
            outputs_folder = inputs_folder.replace("inputs", "outputs")

        if self.run_async_inference:
            logger.debug(f"{event_id} - ASYNC inference run.")
            process_image_data = {
                "event_id": event_id,
                "images": images,
                "image_urls": image_urls,
                "details": {
                    "space_id": input_data.get("space_id"),
                    "project_id": input_data.get("project_id"),
                },
                "inputs_folder": inputs_folder,
                "outputs_folder": outputs_folder,
                "input_data": input_data,
            }
            thread = DestroyableThread(
                runnable_func=self._run_process_images_async,
                runnable_kwargs=process_image_data,
                name=f"Thread-{event_id}",
            )
            thread.start()
            inference_output = copy.deepcopy(input_data)
            inference_output.update(
                {
                    "model_id": self.model_id,
                    "status": "IN_PROGRESS",
                }
            )
            logger.debug(f"{event_id} - ASYNC inference run started.")
        else:
            logger.debug(f"{event_id} - SYNC inference run.")
            inference_output = self.process_images(
                event_id=event_id,
                images=images,
                image_urls=image_urls,
                inputs_folder=inputs_folder,
                outputs_folder=outputs_folder,
                input_data=input_data,
            )

        if "predicted_layers" in input_data:
            output = inference_output.get("output")
            inference_output = {
                "predicted_layers": output,
                "event_id": event_id,
                "model_id": self.model_id,
                "rgb_constant_multiply": input_data.get("rgb_constant_multiply"),
                "data_type": input_data.get("data_type"),
                "asset_categories": input_data.get("asset_categories"),
                "base_style": input_data.get("base_style"),
                "asset_style": input_data.get("asset_style"),
                "request_type": input_data.get("request_type"),
                "output_url": input_data.get("output_url"),
                "bbox": input_data.get("bbox"),
                "status": "COMPLETED",
            }

        return ImageResult(text=str(inference_output))

    @emit_prometheus_metrics(
        [
            (
                lambda labels: RUNNING_GAUGE_METRIC.labels(
                    **{**labels, "step": "save"}
                ).track_inprogress(),
                RUNNING_GAUGE_LABELS,
            ),
            (
                lambda labels: DURATION_SUMMARY_METRIC.labels(
                    **{**labels, "step": "save"}
                ).time(),
                DURATION_SUMMARY_LABELS,
            ),
        ]
    )
    def save(self, artifact_path, *args, **kwargs):
        block_saver = ModuleSaver(self, model_path=artifact_path)

        # Extract object to be saved
        with block_saver:
            block_saver.update_config({"artifact": "models"})
            self.model.save(artifact_path)

    @classmethod
    def load(cls, artifact_path: str):
        """Load a caikit model
        Args:
            artifact_path (str): Path to caikit model.
        Returns:
            cls(model): loaded model
        """
        # Read config file
        logger.debug(f"Artifact path: {artifact_path}")
        config = ModuleConfig.load(artifact_path)
        config_path = os.path.join(artifact_path, config.config_path)
        weights_path = os.path.join(artifact_path, config.checkpoint_file)
        predict_dataset_bands = config.bands
        logger.debug(f"config path: {config_path}")
        logger.debug(f"pretrained_model_path: {weights_path}")
        logger.debug(f"predict_dataset_bands: {predict_dataset_bands}")
        logger.debug("*********LOAD**********")

        try:
            # Replace grep in config to fix Regression issue:
            # GenericNonGeoPixelwiseRegressionDataModule.predict_dataset has length 0
            logger.debug(f"Updating the config img_grep")
            update_grep_config_in_file(config_path = config_path, new_img_pattern = "*.tif*")
            # Convert the old checkpoints before loading
            converted_weights_path = convert_old_checkpoints(weights_path)
            model = LightningInferenceModel.from_config(
                config_path=config_path,
                checkpoint_path=converted_weights_path,
                # predict_dataset_bands=predict_dataset_bands,
            )
            logger.debug("*********MODEL LOADED**********")

        except:
            logger.warning(
                "Something went wrong with the config. Expected Terratorch config."
            )
            raise

        return cls(model)
