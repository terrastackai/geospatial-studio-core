# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

"""Setup to be able to build a wheel for this extension module."""

import os

import setuptools
import setuptools.command.build_py
import yaml

from scripts.development.locutils import CONFIG_PATH, EXTENSION_NAME, PROJECT_ROOT_DIR

with open(CONFIG_PATH, "r", encoding="utf8") as config_handle:
    extension_config = yaml.safe_load(config_handle)

# This is what we were doing before; let's figure out something better for versioning before release
# lib_version = extension_config["version"]

# We need to pass this in somehow by inferring it off of git tags, but we are leaving that for a separate PR
lib_version = os.getenv("COMPONENT_VERSION")
if lib_version is None:
    raise RuntimeError(
        "No version found; set the environment variable COMPONENT_VERSION"
    )

# read requirements from file
with open(os.path.join(PROJECT_ROOT_DIR, "requirements.txt")) as filehandle:
    requirements = list(map(str.strip, filehandle.read().splitlines()))
    # Remove --extra index line, as its not parsable by setup
    requirements = [
        name for name in requirements if not name.startswith("--extra-index-url")
    ]

if __name__ == "__main__":

    # Import caikit.core and the extension to register the dataobjects

    # First Party
    import caikit.core

    # Local
    # Render the dataobject protos
    interface_dir = os.path.join(".", EXTENSION_NAME, "data_model", "interfaces")
    caikit.core.data_model.render_dataobject_protos(interface_dir)

    setuptools.setup(
        name=EXTENSION_NAME,
        author="IBM",
        version=lib_version,
        license="Copyright IBM 2023 -- All rights reserved.",
        description="GFM",
        install_requires=requirements,
        packages=setuptools.find_packages(include=f"{EXTENSION_NAME}*"),
        package_data={
            f"{EXTENSION_NAME}.data_model.interfaces": ["*.proto"],
            f"{EXTENSION_NAME}.config": ["config.yml"],
        },
    )
