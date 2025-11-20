# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import alog
from caikit.core.toolkit import error_handler

log = alog.use_channel("CONFIGVALIDATR")
error = error_handler.get(log)

# Cerberus schema that we expect the project to follow
config_schema = {
    "artifactory_base_path": {"type": "string", "required": True},
    "description": {"type": "string", "required": True},
    "tags": {"type": ["string", "list"], "required": False},
    "maintainers": {"type": "list", "schema": {"type": "string"}, "required": True},
    "code_samples": {"type": "list", "schema": {"type": "string"}, "required": False},
    "enable_code_sample_execution_tests": {"type": "boolean", "required": False},
}

"""
extension_config_validator = cerberus.Validator(config_schema)
# Allow developers to add additional configurations if they like; all
# that matters is that what the skeleton already defines is well upheld
extension_config_validator.allow_unknown = True
"""


def validate_extension_config(extension_config):
    """Validate the extension config against our cerberus schema.

    Args:
        extension_config: dict
            Extension schema against which we will validate our extension.


    extension_config_validator.validate(extension_config)
    conf_errs = extension_config_validator.errors
    if conf_errs:
        error(
            "<EXT12345678E>",
            ValueError("Extension config validation failed with errors: {}", conf_errs),
        )
    """
    pass
