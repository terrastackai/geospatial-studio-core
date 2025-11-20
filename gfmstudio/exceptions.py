# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


class CustomBaseException(Exception):
    """Base exception for all custom exceptions in this project."""

    def __init__(self, message, details=None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message} - Additional details: {self.details}"
        return self.message


class ModelDeploymentError(CustomBaseException):
    """Raised when there is an error in deploying the model."""

    pass


class PresignedLinkExpired(CustomBaseException):
    """Raised when a presigned link has expired"""

    pass
