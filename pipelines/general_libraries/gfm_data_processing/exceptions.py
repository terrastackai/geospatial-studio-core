# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


class GfmDataPipelineException(Exception):
    def __init__(self, message, error_code=None, event_id=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.event_id = event_id

    def __str__(self):
        return f"{self.event_id}: ({self.error_code}) {self.message}"


class GfmDataProcessingException(Exception):
    def __init__(self, message, error_code=None, event_id=None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.event_id = event_id

    def __str__(self):
        return f"{self.event_id}: ({self.error_code}) {self.message}"
