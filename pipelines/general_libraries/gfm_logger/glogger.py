# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging
import os
import datetime


def _patch_logging_for_immediate_flush(log_file_path):
    """
    Monkey-patch logging to write all logs immediately to file.
    This intercepts all logging calls globally to ensure real-time log visibility.
    """
    if hasattr(logging.Logger, '_patched_for_flush'):
        return
    
    log_file = open(log_file_path, 'a', buffering=1)
    
    # Store original logging methods
    original_methods = {
        'info': logging.Logger.info,
        'debug': logging.Logger.debug,
        'warning': logging.Logger.warning,
        'error': logging.Logger.error,
        'critical': logging.Logger.critical,
    }
    
    def create_patched_method(original_method, level_name):
        """Create a patched method that writes to file immediately."""
        def wrapper(self, msg, *args, **kwargs):
            original_method(self, msg, *args, **kwargs)
            try:
                timestamp = datetime.datetime.now().isoformat()
                if args:
                    try:
                        msg = msg % args
                    except (TypeError, ValueError):
                        pass
                log_line = f"{timestamp} - {self.name} - {level_name} - {msg}\n"
                log_file.write(log_line)
                log_file.flush()
                os.fsync(log_file.fileno())
            except Exception:
                pass
        return wrapper
    
    for level_name, original_method in original_methods.items():
        setattr(logging.Logger, level_name, create_patched_method(original_method, level_name.upper()))
    
    logging.Logger._patched_for_flush = True


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
        stdout_handler.setLevel(logging.ERROR)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
        
        _patch_logging_for_immediate_flush(stdout_log)
    else:
        handler = logging.StreamHandler()
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
