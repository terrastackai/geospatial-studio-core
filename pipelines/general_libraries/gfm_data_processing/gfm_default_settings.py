# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


#
# Default setting which are set when not provided in the inference request

default_resolution = {
    "hlsl30": 30,
    "hlss30": 30,
    "hls": 30,
    "hls-agb": 30,
    "s2": 10,
}

default_scaled_bands = {
    "hlsl30": ["0", "1", "2", "3", "4", "5"],
    "hlss30": ["0", "1", "2", "3", "4", "5"],
    "hls": ["0", "1", "2", "3", "4", "5"],
    "hls-agb": ["0", "1", "2", "3", "4", "5"],
    "s2": ["0", "1", "2", "3", "4", "5"],
}

default_collections = {
    "hlsl30": {
        "HLSL30": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B05",
            "4": "B06",
            "5": "B07",
            "6": "Fmask",
        }
    },
    "hlss30": {
        "HLSS30": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B8A",
            "4": "B11",
            "5": "B12",
            "6": "Fmask",
        }
    },
    "hls": {
        "HLSL30": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B05",
            "4": "B06",
            "5": "B07",
            "6": "Fmask",
        },
        "HLSS30": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B8A",
            "4": "B11",
            "5": "B12",
            "6": "Fmask",
        },
    },
    "hls-agb": {
        "hls-agb-preprocessed": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B05",
            "4": "B06",
            "5": "B07",
        }
    },
    "s2": {
        "ibm-eis-ga-1-esa-sentinel-2-l2a": {
            "0": "B02",
            "1": "B03",
            "2": "B04",
            "3": "B8A",
            "4": "B11",
            "5": "B12",
            "6": "SCL",
        }
    },
}

input_layers_metadata_collection = {
    "input_layers_meta_hls": {
        "cluster": "cluster1",
        "output_type": "Float32",
        "layers": [
            {"layer_id": 51733, "level": 21},
            {"layer_id": 51734, "level": 21},
            {"layer_id": 51735, "level": 21},
            {"layer_id": 51736, "level": 21},
            {"layer_id": 51737, "level": 21},
            {"layer_id": 51738, "level": 21},
            {"layer_id": 51742, "level": 21},
        ],
        "cloud_categories": [2, 4, 8],
    },
    "input_layers_meta_hlsl30": {
        "cluster": "cluster1",
        "output_type": "Float32",
        "layers": [
            {"layer_id": 51733, "level": 21},
            {"layer_id": 51734, "level": 21},
            {"layer_id": 51735, "level": 21},
            {"layer_id": 51736, "level": 21},
            {"layer_id": 51737, "level": 21},
            {"layer_id": 51738, "level": 21},
            {"layer_id": 51742, "level": 21},
        ],
        "cloud_categories": [2, 4, 8],
    },
    "input_layers_meta_hlss30": {
        "cluster": "cluster1",
        "output_type": "Float32",
        "layers": [
            {"layer_id": 51796, "level": 21},
            {"layer_id": 51797, "level": 21},
            {"layer_id": 51798, "level": 21},
            {"layer_id": 51803, "level": 21},
            {"layer_id": 51806, "level": 21},
            {"layer_id": 51807, "level": 21},
            {"layer_id": 51808, "level": 21},
        ],
        "cloud_categories": [2, 4, 8],
    },
    "input_layers_meta_s2": {
        "cluster": "None",
        "output_type": "Int16",
        "layers": [
            {"layer_id": 49680, "level": 23},
            {"layer_id": 49681, "level": 23},
            {"layer_id": 49360, "level": 23},
            {"layer_id": 49685, "level": 21},
            {"layer_id": 49686, "level": 21},
            {"layer_id": 49687, "level": 21},
            {"layer_id": 49362, "level": 22},
        ],
        "cloud_categories": [3, 8, 9],
    },
}
