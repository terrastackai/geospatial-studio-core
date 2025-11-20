# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This script modifies a PyTorch checkpoint file by renaming specific keys
in the state dictionary. It is necessary for model checkpoints created by
older versions of terratorch (<0.99.7)

Usage:
    python tt-compatibility-script.py <path_in>

Arguments:
    path_in: Path to the input checkpoint file (.ckpt) that needs to be modified.

Output:
    A new checkpoint file is saved in the same directory as the input file,
    with '_modified' appended to the filename before the file extension.

Example:
    Input:  /app/models/sample.ckpt
    Output: /app/models/sample_modified.ckpt
"""

import os
import sys

import torch


def modify_checkpoint(path_in):
    # Generate path_out by inserting '_modified' before the file extension
    base, ext = os.path.splitext(path_in)
    path_out = f"{base}_modified{ext}"

    state_dict = torch.load(path_in, map_location=torch.device("cpu"))
    state_dict_renamed = {}

    for k, v in state_dict.items():
        if k == "state_dict":
            state_dict_renamed[k] = {}
            for k1, v1 in v.items():
                if "model.encoder." in k1:
                    state_dict_renamed[k][
                        k1.replace("model.encoder.", "model.encoder._timm_module.")
                    ] = v1
                else:
                    state_dict_renamed[k][k1] = v1
        else:
            state_dict_renamed[k] = v

    torch.save(state_dict_renamed, path_out)
    print(f"Modified checkpoint saved to: {path_out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_in>")
        sys.exit(1)

    path_in = sys.argv[1]
    modify_checkpoint(path_in)
