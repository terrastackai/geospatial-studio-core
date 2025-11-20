# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""Sample extension library for Watson NLP"""

# Import subpackages
from . import blocks, config, data_model

# Bring lib_config to the top level of the package for easy access
from .config import *
from .data_model import *
