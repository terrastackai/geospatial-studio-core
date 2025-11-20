# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
from pathlib import Path

import caikit.core
from caikit.config import configure
from caikit.runtime.service_factory import ServicePackageFactory

from .config_validator import validate_extension_config

# The name for an extension is simply the name of the directory containing its config dir.
extension_name = Path(__file__).parent.parent.name

# Create the lib config & patch the extension name into the Artifactory basepath
configure(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml"),
)

inference_service = ServicePackageFactory().get_service_package(
    ServicePackageFactory.ServiceType.INFERENCE,
)

# lib_config.artifactory_base_path = (
#     os.path.join("https://na.artifactory.swg-devops.com/artifactory/api/storage/wcp-nlu-team-wnlp-extensions-models-generic-local", extension_name) + os.sep
# )

# # Cast the aconfig extension config to a dictionary and validate it through cerberus.
# validate_extension_config(dict(lib_config))


#################### Helpers that are nice to expose at the top level of the package
# MODEL_CATALOG = caikit.core.catalog.ModelCatalog(
#     {}, lib_config.library_version, lib_config.artifactory_base_path
# )
# RESOURCE_CATALOG = caikit.core.catalog.ResourceCatalog(
#     {}, lib_config.library_version, lib_config.artifactory_base_path
# )
# WORKFLOW_CATALOG = caikit.core.catalog.WorkflowCatalog(
#     {}, lib_config.library_version, lib_config.artifactory_base_path
# )

# aliases helpers for users
# get_models = MODEL_CATALOG.get_models
# get_alias_models = MODEL_CATALOG.get_alias_models
# get_latest_models = MODEL_CATALOG.get_latest_models
# get_resources = RESOURCE_CATALOG.get_resources
# get_alias_resources = RESOURCE_CATALOG.get_alias_resources
# get_latest_resources = RESOURCE_CATALOG.get_latest_resources
# get_workflows = WORKFLOW_CATALOG.get_workflows

# MODEL_MANAGER = caikit.core.ModelManager(
#     lib_config.artifactory_base_path, MODEL_CATALOG, RESOURCE_CATALOG, WORKFLOW_CATALOG
# )

# download = MODEL_MANAGER.download
# extract = MODEL_MANAGER.extract
# fetch = MODEL_MANAGER.fetch
# load = MODEL_MANAGER.load
# download_and_load = MODEL_MANAGER.download_and_load
# resolve_and_load = MODEL_MANAGER.resolve_and_load
