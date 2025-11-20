# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import re
from pathlib import Path

# Get the root of this project directory
project_root_path = Path(__file__).parent.parent.parent.absolute()
# Relative directories within our project that we explicitly want to exclude from our config search
skip_dirs = [project_root_path / "build"]
config_re = re.compile(r"\/([^\/]*)\/config\/config.yml")


def is_skipped_directory(dir_path):
    """Check if a provided path is a subdirectory of one of our skipped directories.

    Args:
        dir_path: pathlib.PosixPath
            Path object to be checked against skipdirs.

    Returns:
        bool
            True if the dir_path is a subdirectory of any skipped directories, False otherwise.
    """
    if "site-packages" in str(dir_path):
        return True

    for skip_dir in skip_dirs:
        root_parents = dir_path.parents
        if skip_dir in root_parents:
            return True
    return False


def _infer_project_config_path():
    """From the overall project root, do a best-effort-search to locate the config for this
    project. This is accomplished by looking for a file named config.yml in a directory named
    config excluding special directories where the config might be copied at build time.

    Returns:
        str
            Path to config file.
    """
    config_path = None
    for root, _, files in os.walk(project_root_path):
        # Check if this root is in our skip_dirs; if so, skip the files within it
        if is_skipped_directory(Path(root)):
            continue
        # Otherwise consider each file in this subdirectory
        for file in files:
            # Check if our file path matches our config regex
            file_path = os.path.join(root, file)
            path_match = config_re.search(file_path)
            if path_match:
                # If this is the first match, save it
                if config_path is None:
                    config_path = file_path
                # If not, explode; it doesn't matter if these are the same. We need to know the
                # ground truth config location as an anchorpoint in the source code dir in the
                # rest of the project for building our wheel properly.
                else:
                    raise FileNotFoundError(
                        "Unable to correctly infer config due to "
                        "multiple matches at paths:"
                        ": \n\t{}\n\t{}".format(config_path, file_path)
                    )
    # Similarly, if we found nothing, throw a config discovery error
    if not config_path:
        raise RuntimeError(
            "Unable to infer project name; failed to match config regex: {}".format(
                config_re.pattern
            )
        )
    return config_path


def _infer_extension_name(config_path):
    """Given the path to the project config, provide the project root based on the well-defined
    structure of the project. This is useful for things like compiling the project byte code when
    building the wheel, and so on.

    config_path: str
        Path to config file for the extension.

    Returns:
        str
            Project src folder path.
    """
    # We expect config to be one level down for a well-defined watson nlp extension
    lib_path = Path(config_path).parent.parent
    # Ensure that the inferred lib path is still below the overall project
    if project_root_path not in lib_path.parents:
        raise ValueError(
            "Inferred library source root: [{}]".format(lib_path),
            "is not a subdirectory of overall project root: [{}]".format(
                project_root_path
            ),
        )
    return str(lib_path).split(os.sep)[-1]


# Path to the root of the project
PROJECT_ROOT_DIR = str(project_root_path)
# Path to project config file, which contains details like project name, etc
CONFIG_PATH = _infer_project_config_path()
# Name of the src subdirectory inferred above, which is the name of our extension
EXTENSION_NAME = _infer_extension_name(CONFIG_PATH)
# Place where the project source code lives
PROJECT_SRC_DIR = str(project_root_path / EXTENSION_NAME)

print(
    ""
    + PROJECT_ROOT_DIR
    + ","
    + CONFIG_PATH
    + ","
    + EXTENSION_NAME
    + ","
    + PROJECT_SRC_DIR
)
