# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

import collections
import datetime
import glob
import json
import logging
import os
import re
import shutil
import ssl
import sys
import tarfile
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from itertools import chain
from multiprocessing import Pool, Process, Queue, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import boto3
import humanize
import numpy as np
import rasterio
import requests
import wget
from botocore.client import Config
from botocore.exceptions import ClientError
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles
from sklearn.model_selection import train_test_split
from terrakit.chip.tiling import chip_and_label_data
from terrakit.download.download_data import download_data
from terrakit.transform.labels import process_labels
from tqdm import tqdm

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Performance tuning constants
MAX_UPLOAD_WORKERS = int(os.getenv("MAX_UPLOAD_WORKERS", "10"))
MAX_COG_WORKERS = int(os.getenv("MAX_COG_WORKERS", str(max(1, cpu_count() - 1))))
CHUNK_SIZE = int(os.getenv("RASTER_CHUNK_SIZE", "1024"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "29"))

# PVC mount path for direct file storage
COS_PVC_MOUNT = os.getenv("COS_PVC_MOUNT", "")
USE_PVC_STORAGE = bool(COS_PVC_MOUNT and os.path.exists(COS_PVC_MOUNT))

# Default split weights
DEFAULT_SPLIT_WEIGHTS = (0.6, 0.2, 0.2)  # train, test, val

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================


class OnboardingStatus(Enum):
    """Enumeration for onboarding status."""

    ONBOARDING = "Onboarding"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"


class ErrorCode(Enum):
    """Enumeration for error codes."""

    SUCCESS = "0000"
    DOWNLOAD_ERROR = "0001"
    FILE_SORT_ERROR = "0002"
    IMAGE_SIZE_ERROR = "0003"
    FILE_STEM_ERROR = "0004"
    SPLIT_ERROR = "0005"
    SPLIT_UPLOAD_ERROR = "0006"
    TRAINING_PARAMS_ERROR = "0007"
    COG_VALIDATION_ERROR = "0008"
    IMAGE_UPLOAD_ERROR = "0009"
    ONBOARDING_DETAILS_ERROR = "0010"
    PROPERTIES_SAVE_ERROR = "0011"
    FILE_SAVE_ERROR = "0012"
    DATASET_PREP_ERROR = "0013"


@dataclass
class ErrorInfo:
    """Data class for error information."""

    code: str = ErrorCode.SUCCESS.value
    message: str = "N/A"

    def set_error(self, code: ErrorCode, message: str) -> None:
        """Set error code and message."""
        self.code = code.value
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"code": self.code, "message": self.message}


@dataclass
class OnboardingConfig:
    """Configuration for dataset onboarding."""

    dataset_id: str
    dataset_url: str
    label_suffix: str
    data_sources: List[Dict[str, Any]]
    onboarding_options: Dict[str, Any]
    df_api_route: str
    df_api_key: str
    dataset_bucket: str
    working_path: str
    split_weights: Tuple[float, float, float] = DEFAULT_SPLIT_WEIGHTS

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate configuration parameters.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate dataset_id
        if not self.dataset_id or not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")

        # Validate dataset_url
        if not self.dataset_url or not self.dataset_url.strip():
            raise ValueError("dataset_url cannot be empty")

        # Validate label_suffix
        if not self.label_suffix or not self.label_suffix.startswith("."):
            raise ValueError("label_suffix must start with a dot (e.g., '.mask.tif')")

        # Validate data_sources
        if not self.data_sources:
            raise ValueError("data_sources cannot be empty")

        for idx, source in enumerate(self.data_sources):
            if "modality_tag" not in source:
                raise ValueError(f"data_sources[{idx}] missing 'modality_tag'")
            if "file_suffix" not in source:
                raise ValueError(f"data_sources[{idx}] missing 'file_suffix'")
            if "bands" not in source:
                raise ValueError(f"data_sources[{idx}] missing 'bands'")
            if not isinstance(source["bands"], list):
                raise ValueError(f"data_sources[{idx}]['bands'] must be a list")

        # Validate split_weights
        if len(self.split_weights) != 3:
            raise ValueError(
                "split_weights must have exactly 3 values (train, test, val)"
            )

        if not np.isclose(sum(self.split_weights), 1.0):
            raise ValueError(
                f"split_weights must sum to 1.0, got {sum(self.split_weights)}"
            )

        if any(w <= 0 or w >= 1 for w in self.split_weights):
            raise ValueError("Each split weight must be between 0 and 1 (exclusive)")

    @classmethod
    def from_environment(cls) -> "OnboardingConfig":
        """
        Create configuration from environment variables.

        Returns:
            OnboardingConfig instance

        Raises:
            ValueError: If required environment variables are missing or invalid
        """
        # Required environment variables
        dataset_id = os.getenv("dataset_id")
        if not dataset_id:
            raise ValueError("Environment variable 'dataset_id' is required")

        dataset_url = os.getenv("dataset_url")
        if not dataset_url:
            raise ValueError("Environment variable 'dataset_url' is required")

        # Optional with defaults
        label_suffix = os.getenv("label_suffix", ".mask.tif")
        data_sources_str = os.getenv("data_sources", "[]")
        onboarding_options_str = os.getenv("onboarding_options", "{}")
        df_api_route = os.getenv(
            "df_api_route",
            "https://geoft-dataset-factory-api-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/",
        )
        df_api_key = os.getenv("DF_APIKEY", "")
        dataset_bucket = os.getenv("DATA_BUCKET", "geoft-service-datasets")

        # Parse JSON strings with better error handling
        try:
            data_sources = json.loads(data_sources_str)
            if not isinstance(data_sources, list):
                raise ValueError("data_sources must be a JSON array")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in data_sources: {e}") from e

        try:
            onboarding_options = json.loads(onboarding_options_str)
            if not isinstance(onboarding_options, dict):
                raise ValueError("onboarding_options must be a JSON object")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in onboarding_options: {e}") from e

        working_path = f"/data/{dataset_id}"

        return cls(
            dataset_id=dataset_id,
            dataset_url=dataset_url,
            label_suffix=label_suffix,
            data_sources=data_sources,
            onboarding_options=onboarding_options,
            df_api_route=df_api_route,
            df_api_key=df_api_key,
            dataset_bucket=dataset_bucket,
            working_path=working_path,
        )


@dataclass
class TrainingParams:
    """Training parameters for a single modality."""

    modality_tag: str
    norm_means: List[float]
    norm_stds: List[float]
    bands: List[int]
    file_suffix: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "modality_tag": self.modality_tag,
            "norm_means": self.norm_means,
            "norm_stds": self.norm_stds,
            "bands": self.bands,
            "file_suffix": self.file_suffix,
        }


# ============================================================================
# LOGGING SETUP
# ============================================================================


def setup_logging() -> logging.Logger:
    """
    Configure logging with proper formatting and level.

    Returns:
        Configured logger instance
    """
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()

    # Create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create console handler with formatting
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s (%(filename)s:%(lineno)s)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Initialize logger
logger = setup_logging()


# ============================================================================
# OBJECT STORAGE CLIENT
# ============================================================================


class S3ClientManager:
    """Manager for S3 client with connection pooling and context management."""

    def __init__(self, max_pool_connections: int = MAX_UPLOAD_WORKERS * 2):
        """Initialize S3 client manager."""
        self.max_pool_connections = max_pool_connections
        self._client = None

    @property
    def client(self):
        """Get or create S3 client with lazy initialization."""
        if self._client is None:
            self._client = boto3.client(
                "s3",
                endpoint_url=os.getenv("OBJECT_STORAGE_ENDPOINT", ""),
                aws_access_key_id=os.getenv("OBJECT_STORAGE_KEY_ID", ""),
                aws_secret_access_key=os.getenv("OBJECT_STORAGE_SEC_KEY", ""),
                config=Config(
                    signature_version="s3v4",
                    max_pool_connections=self.max_pool_connections,
                ),
                region_name=os.getenv("OBJECT_STORAGE_REGION", ""),
            )
        return self._client

    def __enter__(self):
        """Context manager entry."""
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        if self._client is not None:
            self._client.close()
            self._client = None


# ============================================================================
# API NOTIFICATION
# ============================================================================


class DatasetFactoryNotifier:
    """Handler for notifying the Dataset Factory API about onboarding status."""

    def __init__(self, api_route: str, api_key: str):
        """Initialize notifier."""
        self.api_route = api_route
        self.api_key = api_key
        self.webhooks_url = (
            api_route if "notifications" in api_route else f"{api_route}v2/webhooks"
        )
        self.headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        }

    def notify(self, onboarding_details: Dict[str, Any]) -> bool:
        """Send notification to Dataset Factory API."""
        logger.info("Notifying Dataset Factory API of onboarding status")

        event_data = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "detail": onboarding_details,
            "detail_type": "FT:Data:Onboarding",
            "source": "com.ibm.dataset-factory-onboarding",
        }

        # Update event type for finished states
        if "notifications" in self.webhooks_url and onboarding_details.get(
            "status"
        ) in [
            OnboardingStatus.SUCCEEDED.value,
            OnboardingStatus.FAILED.value,
        ]:
            event_data["detail_type"] = "FT:Data:Finished"

        try:
            response = requests.post(
                url=self.webhooks_url,
                headers=self.headers,
                json=event_data,
                timeout=REQUEST_TIMEOUT,
                verify=False,
            )

            if response.status_code not in (200, 201):
                logger.error(
                    f"Failed to send notification. Status: {response.status_code}, "
                    f"Response: {response.text}"
                )
                return False

            logger.info("Successfully sent notification to Dataset Factory API")
            return True

        except requests.exceptions.RequestException as ex:
            logger.error(f"Failed to send notification due to request error: {ex}")
            return False
        except Exception as ex:
            logger.error(f"Unexpected error sending notification: {ex}", exc_info=True)
            return False


# ============================================================================
# ONBOARDING DETAILS BUILDER
# ============================================================================


class OnboardingDetailsBuilder:
    """Builder for constructing onboarding details dictionary."""

    def __init__(self, dataset_id: str):
        """Initialize builder."""
        self.dataset_id = dataset_id
        self.details: Dict[str, Any] = {
            "dataset_id": dataset_id,
            "status": OnboardingStatus.ONBOARDING.value,
            "error_code": ErrorCode.SUCCESS.value,
            "error_message": "N/A",
        }

    def set_status(self, status: OnboardingStatus) -> "OnboardingDetailsBuilder":
        """Set onboarding status."""
        self.details["status"] = status.value
        return self

    def set_error(self, error_info: ErrorInfo) -> "OnboardingDetailsBuilder":
        """Set error information."""
        self.details["error_code"] = error_info.code
        self.details["error_message"] = error_info.message
        return self

    def set_size(self, size: str) -> "OnboardingDetailsBuilder":
        """Set dataset size."""
        self.details["size"] = size
        return self

    def set_training_params(
        self, training_params_list: List[TrainingParams]
    ) -> "OnboardingDetailsBuilder":
        """Set training parameters for all modalities."""
        training_params: Dict[str, Any] = {}

        # Add split file paths
        stages = ["train", "test", "val"]
        for stage in stages:
            training_params[f"{stage}_split_path"] = (
                f"/{self.dataset_id}/split_files/{stage}_data.txt"
            )

        # Add modality-specific parameters
        for params in training_params_list:
            modality_tag = params.modality_tag
            training_params[modality_tag] = {
                "norm_means": params.norm_means,
                "norm_stds": params.norm_stds,
                "bands": params.bands,
                "file_suffix": f"*{params.file_suffix}",
            }

            # Add data and label directories for each stage
            for stage in stages:
                training_params[modality_tag][
                    f"{stage}_data_dir"
                ] = f"/{self.dataset_id}/training_data/{modality_tag}/"
                training_params[modality_tag][
                    f"{stage}_labels_dir"
                ] = f"/{self.dataset_id}/labels/"

        self.details["training_params"] = training_params
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the onboarding details dictionary."""
        return self.details


# ============================================================================
# FILE OPERATIONS
# ============================================================================


class FileOperations:
    """Utility class for file operations."""

    @staticmethod
    def extract_file_stems(filepaths: List[str], suffix: str) -> List[str]:
        """Extract file stems (names without suffix) from file paths."""
        logger.info(f"Extracting file stems for {len(filepaths)} files")
        try:
            return [Path(filepath).name.replace(suffix, "") for filepath in filepaths]
        except Exception as e:
            logger.error(f"Error extracting file stems: {e}", exc_info=True)
            raise ValueError(f"Failed to extract file stems: {e}") from e

    @staticmethod
    def find_and_sort_files(filepath: str, suffix: str) -> List[str]:
        """Find and sort files with a specific suffix."""
        logger.info(f"Searching for files with suffix '{suffix}' in {filepath}")
        try:
            pattern = f"{filepath}/**/*{suffix}"
            files = sorted(glob.glob(pattern, recursive=True))
            logger.info(f"Found {len(files)} files")
            return files
        except Exception as e:
            logger.error(f"Error finding files: {e}", exc_info=True)
            raise ValueError(f"Failed to find files: {e}") from e

    @staticmethod
    def calculate_total_size(files: List[str]) -> str:
        """Calculate total size of files in human-readable format."""
        logger.info(f"Calculating total size for {len(files)} files")
        total_size = sum(os.path.getsize(f) for f in files if os.path.exists(f))
        return humanize.naturalsize(total_size)

    @staticmethod
    def verify_image_dimensions(image_paths: List[str]) -> None:
        """Verify that all images have consistent dimensions."""
        logger.info(f"Verifying dimensions for {len(image_paths)} images")

        try:
            all_sizes = []
            for image_path in image_paths:
                with rasterio.open(image_path) as img:
                    all_sizes.append(img.shape)

            unique_sizes = collections.Counter(all_sizes)

            if len(unique_sizes) > 1:
                majority_size, _ = unique_sizes.most_common()[0]
                outliers = [
                    Path(image_paths[i]).name
                    for i, size in enumerate(all_sizes)
                    if size != majority_size
                ]

                # Log first 10 outliers
                outlier_sample = outliers[:10]
                logger.error(
                    f"Found {len(outliers)} images with inconsistent dimensions"
                )
                logger.error(f"Sample outliers: {outlier_sample}")

                raise ValueError(
                    f"Inconsistent image dimensions detected. {len(outliers)} images "
                    f"do not match the majority size {majority_size}. "
                    f"Sample: {outlier_sample}. All images must have the same dimensions."
                )

            logger.info(f"All images have consistent dimensions: {all_sizes[0]}")

        except rasterio.errors.RasterioIOError as e:
            logger.error(f"Error reading image file: {e}", exc_info=True)
            raise ValueError(f"Failed to read image file: {e}") from e
        except Exception as e:
            logger.error(f"Error verifying image dimensions: {e}", exc_info=True)
            raise


# ============================================================================
# DATASET DOWNLOAD
# ============================================================================


class DatasetDownloader:
    """Handler for downloading and extracting datasets."""

    @staticmethod
    def download_and_extract(source_url: str, destination: str) -> None:
        """Download and extract dataset from URL. Supports ZIP and TAR archives."""
        logger.info(f"Downloading dataset from {source_url}")

        try:
            # Create destination directory
            os.makedirs(destination, exist_ok=True)

            # Disable SSL verification for download
            ssl._create_default_https_context = ssl._create_unverified_context

            # Download file
            filename = wget.download(source_url, out=destination)
            logger.info(f"\nDownloaded file: {filename}")

            # Extract based on file type
            if zipfile.is_zipfile(filename):
                logger.info("Extracting ZIP archive")
                with zipfile.ZipFile(filename, "r") as zip_ref:
                    zip_ref.extractall(destination)
                os.remove(filename)  # Clean up archive

            elif tarfile.is_tarfile(filename):
                logger.info("Extracting TAR archive")
                with tarfile.open(filename) as tar:
                    tar.extractall(destination)
                os.remove(filename)  # Clean up archive

            else:
                raise ValueError(
                    f"Unsupported file type: {filename}. "
                    "Please provide a ZIP or TAR archive."
                )

            logger.info(f"Successfully extracted dataset to {destination}")

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}", exc_info=True)
            raise ValueError(f"Failed to download dataset: {e}") from e


# ============================================================================
# COG PROCESSING
# ============================================================================


class COGProcessor:
    """Handler for Cloud Optimized GeoTIFF (COG) validation and conversion."""

    @staticmethod
    def process_single_file(filepath: str, dest_dir: str) -> Tuple[bool, Optional[str]]:
        """Process a single file for COG validation and conversion."""
        try:
            filename = Path(filepath).name

            # Validate COG format
            is_valid, errors, warnings = cog_validate(filepath)

            if not is_valid:
                if errors:
                    logger.debug(f"COG validation errors for {filename}: {errors}")

                logger.info(f"Converting {filename} to COG format")

                # Convert to COG using rio-cogeo
                output_path = f"{filepath}.cog.tif"
                cog_translate(
                    filepath,
                    output_path,
                    cog_profiles.get("lzw"),
                    in_memory=False,
                )

                # Replace original with COG version
                os.remove(filepath)
                os.rename(output_path, filepath)

            # Move to destination
            dest_path = os.path.join(dest_dir, filename)
            shutil.move(filepath, dest_path)

            return (True, None)

        except Exception as e:
            error_msg = f"Error processing {filepath}: {str(e)}"
            logger.error(error_msg)
            return (False, error_msg)

    @staticmethod
    def process_batch(filepaths: List[str], dest_dir: str) -> None:
        """Process multiple files for COG validation using multiprocessing."""
        logger.info(f"Processing {len(filepaths)} files for COG validation")

        try:
            os.makedirs(dest_dir, exist_ok=True)

            # Use multiprocessing pool for parallel processing
            with Pool(processes=MAX_COG_WORKERS) as pool:
                results = []
                for filepath in filepaths:
                    result = pool.apply_async(
                        COGProcessor.process_single_file, (filepath, dest_dir)
                    )
                    results.append((filepath, result))

                # Collect results with progress bar
                failed_files = []
                for filepath, result in tqdm(results, desc="Processing COGs"):
                    success, error_msg = result.get()
                    if not success:
                        failed_files.append((filepath, error_msg))

                if failed_files:
                    logger.error(f"{len(failed_files)} files failed COG processing")
                    for filepath, error_msg in failed_files[:5]:
                        logger.error(error_msg)
                    raise RuntimeError(
                        f"{len(failed_files)} files failed COG processing"
                    )

            logger.info("COG processing completed successfully")

        except Exception as e:
            logger.error(f"Error during COG processing: {e}", exc_info=True)
            raise


# ============================================================================
# STORAGE OPERATIONS
# ============================================================================


class StorageManager:
    """Manager for uploading files to S3 or copying to PVC."""

    def __init__(self, use_pvc: bool = USE_PVC_STORAGE, pvc_mount: str = COS_PVC_MOUNT):
        """Initialize storage manager."""
        self.use_pvc = use_pvc
        self.pvc_mount = pvc_mount
        self.s3_manager: Optional[S3ClientManager] = (
            None if use_pvc else S3ClientManager()
        )

    def _copy_to_pvc(
        self, local_path: str, pvc_path: str
    ) -> Tuple[bool, Optional[str]]:
        """Copy a file to PVC mount."""
        try:
            os.makedirs(os.path.dirname(pvc_path), exist_ok=True)
            shutil.copy2(local_path, pvc_path)
            return (True, None)
        except Exception as e:
            error_msg = f"Failed to copy {local_path} to PVC: {str(e)}"
            return (False, error_msg)

    def _upload_to_s3(
        self, s3_client, local_path: str, bucket: str, key: str
    ) -> Tuple[bool, Optional[str]]:
        """Upload a file to S3."""
        try:
            s3_client.upload_file(local_path, bucket, key)
            return (True, None)
        except ClientError as e:
            error_msg = f"S3 upload failed for {local_path}: {str(e)}"
            return (False, error_msg)
        except Exception as e:
            error_msg = f"Unexpected error uploading {local_path}: {str(e)}"
            return (False, error_msg)

    def upload_file(
        self, local_path: str, bucket: str, key: str
    ) -> Tuple[bool, Optional[str]]:
        """Upload a file to storage (S3 or PVC)."""
        if self.use_pvc:
            pvc_path = os.path.join(self.pvc_mount, key)
            return self._copy_to_pvc(local_path, pvc_path)
        else:
            if self.s3_manager is None:
                return (False, "S3 manager not initialized")
            return self._upload_to_s3(self.s3_manager.client, local_path, bucket, key)

    def upload_batch(
        self,
        files: List[str],
        bucket: str,
        key_prefix: str,
        description: str = "Uploading",
    ) -> None:
        """Upload multiple files in parallel."""
        storage_type = "PVC" if self.use_pvc else "S3"
        logger.info(f"Uploading {len(files)} files to {storage_type}")

        try:
            with ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS) as executor:
                futures = []
                for file_path in files:
                    filename = os.path.basename(file_path)
                    key = f"{key_prefix}/{filename}"
                    future = executor.submit(self.upload_file, file_path, bucket, key)
                    futures.append((file_path, future))

                # Collect results with progress bar
                failed_uploads = []
                for file_path, future in tqdm(futures, desc=description):
                    success, error_msg = future.result()
                    if not success:
                        failed_uploads.append((file_path, error_msg))

                if failed_uploads:
                    logger.error(f"{len(failed_uploads)} uploads failed")
                    for file_path, error_msg in failed_uploads[:5]:
                        logger.error(error_msg)
                    raise RuntimeError(f"{len(failed_uploads)} uploads failed")

            logger.info(f"Successfully uploaded {len(files)} files to {storage_type}")

        except Exception as e:
            logger.error(f"Error during batch upload: {e}", exc_info=True)
            raise


# ============================================================================
# SPLIT FILE GENERATION
# ============================================================================


class SplitFileGenerator:
    """Generator for train/test/val split files."""

    @staticmethod
    def create_splits(
        split_weights: Tuple[float, float, float],
        label_files: List[str],
        image_file_lists: List[List[str]],
    ) -> Tuple[Tuple[List, List], Tuple[List, List], Tuple[List, List]]:
        """Create train/test/val splits for images and labels."""
        logger.info("Creating data splits")

        if not np.isclose(sum(split_weights), 1.0):
            raise ValueError(f"Split weights must sum to 1.0, got {sum(split_weights)}")

        try:
            train_size, test_size, val_size = split_weights
            test_val_size = 1 - train_size

            # Transpose image lists for splitting
            image_array = np.array(image_file_lists).T

            # First split: train vs (test + val)
            x_train, x_test_val, y_train, y_test_val = train_test_split(
                image_array,
                label_files,
                train_size=train_size,
                test_size=test_val_size,
                random_state=0,
            )

            # Second split: test vs val
            val_ratio = val_size / test_val_size
            test_ratio = test_size / test_val_size

            x_test, x_val, y_test, y_val = train_test_split(
                x_test_val,
                y_test_val,
                train_size=val_ratio,
                test_size=test_ratio,
                random_state=0,
            )

            # Transpose back to get lists per modality
            x_train_lists = x_train.T.tolist()
            x_test_lists = x_test.T.tolist()
            x_val_lists = x_val.T.tolist()

            logger.info(
                f"Split sizes - Train: {len(y_train)}, "
                f"Test: {len(y_test)}, Val: {len(y_val)}"
            )

            return (
                (x_train_lists, y_train.tolist()),
                (x_test_lists, y_test.tolist()),
                (x_val_lists, y_val.tolist()),
            )

        except Exception as e:
            logger.error(f"Error creating splits: {e}", exc_info=True)
            raise ValueError(f"Failed to create splits: {e}") from e

    @staticmethod
    def save_split_files(
        output_dir: str,
        label_splits: Tuple[List[str], List[str], List[str]],
        label_suffix: str,
    ) -> None:
        """Save split files to disk."""
        logger.info("Saving split files")

        try:
            split_dir = Path(output_dir) / "split_files"
            split_dir.mkdir(parents=True, exist_ok=True)

            stages = ["train", "test", "val"]
            for stage, file_list in zip(stages, label_splits):
                split_file = split_dir / f"{stage}_data.txt"

                with open(split_file, "w") as f:
                    for label_path in sorted(file_list):
                        stem = Path(label_path).name.replace(label_suffix, "")
                        f.write(f"{stem}\n")

                logger.info(f"Saved {stage} split with {len(file_list)} samples")

        except Exception as e:
            logger.error(f"Error saving split files: {e}", exc_info=True)
            raise


# ============================================================================
# STATISTICS CALCULATION
# ============================================================================


class StatisticsCalculator:
    """Calculator for image statistics (mean and standard deviation)."""

    @staticmethod
    def calculate_mean_std(
        training_images: List[str], bands: List[int], chunk_size: int = CHUNK_SIZE
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Calculate mean and standard deviation using chunked processing."""
        logger.info(f"Calculating statistics for {len(training_images)} images")

        try:
            num_bands = len(bands)

            # Initialize accumulators
            total_sum = np.zeros(num_bands, dtype=np.float64)
            total_sum_sqs = np.zeros(num_bands, dtype=np.float64)
            total_pixels = 0
            processed_count = 0

            for path in tqdm(training_images, desc="Computing statistics"):
                try:
                    with rasterio.open(path) as src:
                        height, width = src.height, src.width

                        # Process in chunks
                        for row_start in range(0, height, chunk_size):
                            row_end = min(row_start + chunk_size, height)

                            for col_start in range(0, width, chunk_size):
                                col_end = min(col_start + chunk_size, width)

                                # Read chunk
                                window = rasterio.windows.Window(
                                    col_start,
                                    row_start,
                                    col_end - col_start,
                                    row_end - row_start,
                                )
                                chunk = src.read(window=window)
                                chunk = chunk[bands, :, :].astype(np.float64)

                                # Clean up invalid values
                                chunk[chunk <= -9999] = np.nan

                                # Skip if all NaN
                                if np.all(np.isnan(chunk)):
                                    continue

                                # Replace NaN with band means for this chunk
                                for i in range(num_bands):
                                    band_data = chunk[i]
                                    if np.any(~np.isnan(band_data)):
                                        band_mean = np.nanmean(band_data)
                                        np.nan_to_num(
                                            band_data, nan=band_mean, copy=False
                                        )

                                # Accumulate statistics
                                chunk_sum = chunk.sum(axis=(1, 2))
                                chunk_sum_sqs = (chunk**2).sum(axis=(1, 2))

                                total_sum += chunk_sum
                                total_sum_sqs += chunk_sum_sqs
                                total_pixels += chunk.shape[1] * chunk.shape[2]

                        processed_count += 1

                except Exception as e:
                    logger.warning(f"Failed to process {path}: {e}")
                    continue

            if processed_count == 0 or total_pixels == 0:
                raise ValueError("No valid images were processed")

            # Calculate final statistics
            mean = total_sum / total_pixels
            variance = (total_sum_sqs / total_pixels) - (mean**2)
            std = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative

            logger.info(f"Processed {processed_count}/{len(training_images)} images")
            logger.info(f"Mean: {mean}")
            logger.info(f"Std: {std}")

            return (mean, std, bands)

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}", exc_info=True)
            raise


# ============================================================================
# MODALITY PROCESSOR
# ============================================================================


class ModalityProcessor:
    """Processor for a single data modality."""

    def __init__(
        self,
        modality_tag: str,
        file_suffix: str,
        bands: List[Dict[str, Any]],
        image_files: List[str],
        working_dir: str,
    ):
        """Initialize modality processor."""
        self.modality_tag = modality_tag
        self.file_suffix = file_suffix
        self.bands = [int(b["index"]) for b in bands]
        self.image_files = image_files
        self.working_dir = working_dir
        self.output_dir = os.path.join(working_dir, "training_data", modality_tag)

    def process(self, train_files: List[str]) -> TrainingParams:
        """Process modality: calculate statistics and convert to COG."""
        logger.info(f"Processing modality: {self.modality_tag}")

        try:
            # Calculate statistics
            mean, std, bands = StatisticsCalculator.calculate_mean_std(
                train_files, self.bands
            )

            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Process COGs
            COGProcessor.process_batch(self.image_files, self.output_dir)

            # Create training parameters
            params = TrainingParams(
                modality_tag=self.modality_tag,
                norm_means=mean.tolist(),
                norm_stds=std.tolist(),
                bands=bands,
                file_suffix=self.file_suffix,
            )

            logger.info(f"Completed processing for modality: {self.modality_tag}")
            return params

        except Exception as e:
            logger.error(
                f"Error processing modality {self.modality_tag}: {e}",
                exc_info=True,
            )
            raise


# ============================================================================
# MAIN ONBOARDING ORCHESTRATOR
# ============================================================================


class DatasetOnboardingOrchestrator:
    """Main orchestrator for dataset onboarding pipeline."""

    def __init__(self, config: OnboardingConfig):
        """Initialize orchestrator."""
        self.config = config
        self.notifier = DatasetFactoryNotifier(config.df_api_route, config.df_api_key)
        self.storage_manager = StorageManager()
        self.error_info = ErrorInfo()

    def run(self) -> None:
        """Execute the complete onboarding pipeline."""
        logger.info("=" * 80)
        logger.info("Starting dataset onboarding pipeline")
        logger.info(f"Dataset ID: {self.config.dataset_id}")
        logger.info("=" * 80)

        # Notify start
        start_details = OnboardingDetailsBuilder(self.config.dataset_id).build()
        self.notifier.notify(start_details)

        try:
            # Step 1: Download dataset
            self._download_dataset()

            # Step 2: Prepare dataset (if needed)
            self._prepare_dataset()

            # Step 3: Find files
            image_file_lists, label_files = self._find_files()

            # Step 4: Validate files
            self._validate_files(image_file_lists, label_files)

            # Step 5: Create splits
            splits = self._create_splits(image_file_lists, label_files)

            # Step 6: Process modalities
            training_params_list = self._process_modalities(image_file_lists, splits)

            # Step 7: Process labels
            self._process_labels(label_files)

            # Step 8: Calculate size
            all_files = list(chain.from_iterable(image_file_lists)) + label_files
            size = FileOperations.calculate_total_size(all_files)

            # Step 9: Save properties
            self._save_properties(training_params_list)

            # Step 10: Build and send success notification
            details = (
                OnboardingDetailsBuilder(self.config.dataset_id)
                .set_status(OnboardingStatus.SUCCEEDED)
                .set_size(size)
                .set_training_params(training_params_list)
                .set_error(self.error_info)
                .build()
            )

            logger.info("=" * 80)
            logger.info("Dataset onboarding completed successfully!")
            logger.info(f"Total size: {size}")
            logger.info("=" * 80)

            self.notifier.notify(details)

        except Exception as e:
            logger.error(f"Onboarding failed: {e}", exc_info=True)

            # Build and send failure notification
            details = (
                OnboardingDetailsBuilder(self.config.dataset_id)
                .set_status(OnboardingStatus.FAILED)
                .set_size("0MB")
                .set_error(self.error_info)
                .build()
            )

            self.notifier.notify(details)
            raise

    def _download_dataset(self) -> None:
        """Download and extract dataset."""
        logger.info("Step 1: Downloading dataset")
        try:
            DatasetDownloader.download_and_extract(
                self.config.dataset_url, self.config.working_path
            )
        except Exception as e:
            self.error_info.set_error(ErrorCode.DOWNLOAD_ERROR, str(e))
            raise

    def _prepare_dataset(self) -> None:
        """Prepare dataset if needed (from labels)."""
        if self.config.onboarding_options.get("from_labels"):
            logger.info("Step 2: Preparing dataset from labels")
            try:
                process_labels(
                    dataset_name=self.config.dataset_id,
                    working_dir=self.config.working_path,
                    labels_folder=self.config.working_path,
                )
                queried_data = download_data(
                    dataset_name=self.config.dataset_id,
                    working_dir=self.config.working_path,
                )
                chip_and_label_data(
                    dataset_name=self.config.dataset_id,
                    working_dir=self.config.working_path,
                    queried_data=queried_data,
                    chip_label_suffix=self.config.label_suffix,
                    keep_files=False,
                )
            except Exception as e:
                self.error_info.set_error(ErrorCode.DATASET_PREP_ERROR, str(e))
                raise

    def _find_files(self) -> Tuple[List[List[str]], List[str]]:
        """
        Find all image and label files.

        Returns:
            Tuple of (image_file_lists, label_files)

        Raises:
            ValueError: If no files are found or file search fails
        """
        logger.info("Step 3: Finding image and label files")
        try:
            image_file_lists = []
            for data_source in self.config.data_sources:
                files = FileOperations.find_and_sort_files(
                    self.config.working_path, data_source["file_suffix"]
                )
                if not files:
                    raise ValueError(
                        f"No files found with suffix '{data_source['file_suffix']}' "
                        f"for modality '{data_source.get('modality_tag', 'unknown')}'"
                    )
                image_file_lists.append(files)

            label_files = FileOperations.find_and_sort_files(
                self.config.working_path, self.config.label_suffix
            )

            if not label_files:
                raise ValueError(
                    f"No label files found with suffix '{self.config.label_suffix}'"
                )

            logger.info(
                f"Found {len(label_files)} label files and "
                f"{sum(len(files) for files in image_file_lists)} image files"
            )
            return image_file_lists, label_files

        except Exception as e:
            self.error_info.set_error(ErrorCode.FILE_SORT_ERROR, str(e))
            raise

    def _validate_files(
        self, image_file_lists: List[List[str]], label_files: List[str]
    ) -> None:
        """
        Validate image and label files for consistency.

        Args:
            image_file_lists: List of image file lists per modality
            label_files: List of label files

        Raises:
            ValueError: If validation fails (dimension mismatch, stem mismatch, etc.)
        """
        logger.info("Step 4: Validating files")
        try:
            # Verify we have files to validate
            if not label_files:
                raise ValueError("No label files to validate")

            if not image_file_lists or not any(image_file_lists):
                raise ValueError("No image files to validate")

            # Verify dimensions
            all_images = list(chain.from_iterable(image_file_lists))
            FileOperations.verify_image_dimensions(all_images + label_files)

            # Verify file stems match
            label_stems = set(
                FileOperations.extract_file_stems(label_files, self.config.label_suffix)
            )

            for idx, image_files in enumerate(image_file_lists):
                # Find matching data source
                data_source = next(
                    (
                        ds
                        for ds in self.config.data_sources
                        if any(f.endswith(ds["file_suffix"]) for f in image_files)
                    ),
                    None,
                )

                if not data_source:
                    raise ValueError(f"Could not find data source for image list {idx}")

                image_stems = set(
                    FileOperations.extract_file_stems(
                        image_files, data_source["file_suffix"]
                    )
                )

                if image_stems != label_stems:
                    missing_in_images = label_stems - image_stems
                    missing_in_labels = image_stems - label_stems

                    error_parts = []
                    if missing_in_images:
                        error_parts.append(
                            f"Labels without images: {list(missing_in_images)[:5]}"
                        )
                    if missing_in_labels:
                        error_parts.append(
                            f"Images without labels: {list(missing_in_labels)[:5]}"
                        )

                    raise ValueError(
                        f"File stem mismatch for modality '{data_source['modality_tag']}'. "
                        + "; ".join(error_parts)
                    )

            logger.info("File validation completed successfully")

        except Exception as e:
            if isinstance(e, ValueError) and "stem" in str(e).lower():
                self.error_info.set_error(ErrorCode.FILE_STEM_ERROR, str(e))
            else:
                self.error_info.set_error(ErrorCode.IMAGE_SIZE_ERROR, str(e))
            raise

    def _create_splits(
        self, image_file_lists: List[List[str]], label_files: List[str]
    ) -> Tuple:
        """
        Create train/test/val splits.

        Args:
            image_file_lists: List of image file lists per modality
            label_files: List of label files

        Returns:
            Tuple of ((train_images, train_labels), (test_images, test_labels),
                     (val_images, val_labels))

        Raises:
            ValueError: If split creation fails
        """
        logger.info("Step 5: Creating data splits")
        try:
            splits = SplitFileGenerator.create_splits(
                self.config.split_weights, label_files, image_file_lists
            )

            # Save split files
            train_pair, test_pair, val_pair = splits
            y_splits = (train_pair[1], test_pair[1], val_pair[1])

            SplitFileGenerator.save_split_files(
                self.config.working_path,
                y_splits,
                self.config.label_suffix,
            )

            return splits

        except Exception as e:
            self.error_info.set_error(ErrorCode.SPLIT_ERROR, str(e))
            raise

    def _process_modalities(
        self, image_file_lists: List[List[str]], splits: Tuple
    ) -> List[TrainingParams]:
        """
        Process all modalities (calculate statistics, convert to COG).

        Args:
            image_file_lists: List of image file lists per modality
            splits: Split data tuple from _create_splits

        Returns:
            List of TrainingParams for each modality

        Raises:
            RuntimeError: If modality processing fails
        """
        logger.info("Step 6: Processing modalities")

        try:
            train_pair, test_pair, val_pair = splits
            training_params_list = []

            # Process each modality
            for idx, data_source in enumerate(self.config.data_sources):
                processor = ModalityProcessor(
                    modality_tag=data_source["modality_tag"],
                    file_suffix=data_source["file_suffix"],
                    bands=data_source["bands"],
                    image_files=image_file_lists[idx],
                    working_dir=self.config.working_path,
                )

                # Use training files for statistics
                train_files = train_pair[0][idx]
                params = processor.process(train_files)
                training_params_list.append(params)

            return training_params_list

        except Exception as e:
            self.error_info.set_error(ErrorCode.TRAINING_PARAMS_ERROR, str(e))
            raise

    def _process_labels(self, label_files: List[str]) -> None:
        """
        Process and convert labels to COG format.

        Args:
            label_files: List of label file paths

        Raises:
            RuntimeError: If label processing fails
        """
        logger.info("Step 7: Processing labels")

        try:
            labels_dir = os.path.join(self.config.working_path, "labels")
            COGProcessor.process_batch(label_files, labels_dir)

        except Exception as e:
            self.error_info.set_error(ErrorCode.COG_VALIDATION_ERROR, str(e))
            raise

    def _save_properties(self, training_params_list: List[TrainingParams]) -> None:
        """
        Save dataset properties to JSON file.

        Args:
            training_params_list: List of training parameters for all modalities

        Raises:
            IOError: If file save fails
        """
        logger.info("Step 8: Saving dataset properties")

        try:
            properties_file = os.path.join(
                self.config.working_path, "dataset_properties.json"
            )

            properties = [params.to_dict() for params in training_params_list]

            with open(properties_file, "w") as f:
                json.dump(properties, f, indent=4)

            logger.info(f"Saved properties to {properties_file}")

        except Exception as e:
            self.error_info.set_error(ErrorCode.PROPERTIES_SAVE_ERROR, str(e))
            raise


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main() -> None:
    """Main entry point for the dataset onboarding pipeline."""
    try:
        # Load configuration from environment
        config = OnboardingConfig.from_environment()

        # Create and run orchestrator
        orchestrator = DatasetOnboardingOrchestrator(config)
        orchestrator.run()

        logger.info("Dataset onboarding pipeline completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Dataset onboarding pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
