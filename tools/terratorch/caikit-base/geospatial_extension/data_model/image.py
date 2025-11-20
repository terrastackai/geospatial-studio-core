# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""Data structures for summary representations"""

# First party
from caikit.core import DataObjectBase
from caikit.core.data_model import dataobject


@dataobject
class ImageResult(DataObjectBase):
    """The result image"""

    text: str
