# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


#
# This routine's purpose is to clean up directories & files under
# the /app/output directory of inference server deployments.
#
import logging
import math
import os
import re
import shutil
import time
from datetime import datetime


def create_logger(path: str):
    logger = logging.getLogger(__name__)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(path)
        fh.setLevel(logging.DEBUG)

        # create console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # add the handlers to logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


logger = create_logger("/tmp/cleanup.log")
DEFAULT_DATA_RETENTION_HRS = 30 * 24  # Default is 30days. Minimum is 1hr(s).
INFERENCE_DATA_RETENTION_HRS = os.getenv(
    "INFERENCE_DATA_RETENTION_HRS", DEFAULT_DATA_RETENTION_HRS
)
try:
    INFERENCE_DATA_RETENTION_HRS = math.ceil(float(INFERENCE_DATA_RETENTION_HRS))
except ValueError:
    logger.warning(
        "ValueError: INFERENCE_DATA_RETENTION_HRS should be int/float."
        f" Using default: {DEFAULT_DATA_RETENTION_HRS}hrs."
    )
    INFERENCE_DATA_RETENTION_HRS = DEFAULT_DATA_RETENTION_HRS
finally:
    DATA_RETENTION_SEC = INFERENCE_DATA_RETENTION_HRS * 60 * 60

now = time.time()


def cleanup_files(path: str, data_retention: int = 86400):
    uuid_re = "^.*[0-9a-fA-F]{8}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{4}\\b-[0-9a-fA-F]{12}.*$"

    for f in os.listdir(path):
        delete_path = os.path.join(path, f)
        if (re.search(uuid_re, f)) and (
            os.stat(delete_path).st_mtime < now - 1 * data_retention
        ):
            try:
                shutil.rmtree(delete_path)
                logger.info("Recursively deleted...%s", f)
            except NotADirectoryError:
                os.remove(delete_path)
                logger.info(f"File removed after {data_retention}sec: {delete_path}")
            except Exception:
                logger.exception("Error removing directory: %s", delete_path)


if __name__ == "__main__":
    logger.info("********************************************************************")
    logger.info("Inference server cleanup started: %s", datetime.fromtimestamp(now))
    logger.info("********************************************************************")

    # Cleanup dangling files after failed inference run.
    cleanup_files(path="/app/output")
    # Cleanup completed files after successful inference run.
    cleanup_files(path="/app/output/completed", data_retention=DATA_RETENTION_SEC)

    logger.info("Inference server cleanup completed.")
