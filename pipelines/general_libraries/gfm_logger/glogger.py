# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import os


def configure_logger(log_level):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    stdout_log = os.environ.get('GFM_STDOUT_LOG')
    stderr_log = os.environ.get('GFM_STDERR_LOG')
    
    # Create a formatter to specify the format of the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    
    if stderr_log:
        stderr_handler = logging.FileHandler(stderr_log, mode='a')
        stderr_handler.setLevel(log_level)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
        
    if stdout_log:
        stdout_handler = logging.FileHandler(stdout_log, mode='a')
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
