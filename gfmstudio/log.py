# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import logging

import structlog

from gfmstudio.config import settings

loglevel = logging.DEBUG if settings.DEBUG else logging.INFO
loglevel_name = logging.getLevelName(loglevel)
logging.getLogger("botocore").setLevel(loglevel)
logging.getLogger("boto3").setLevel(loglevel)
logging.getLogger("urllib3").setLevel(loglevel)
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)
logger = structlog.get_logger()
