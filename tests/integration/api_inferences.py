# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
Example payloads for integration tests (Inference).
Each payload is a Python dict, instead of JSON files.
"""

INFERENCE_WX_WIND_TEXAS = {
    "fine_tuning_id": "sandbox",
    "spatial_domain": {
        "urls": [
            "https://ibm.box.com/shared/static/n09qxmdjl2sdbyrixnd14bkgvkbdzoau.nc"
        ]
    },
    "temporal_domain": [],
    "geoserver_push": [
        {
            "workspace": "geofm",
            "layer_name": "ws10m",
            "display_name": "WxC Wind Speed 10M",
            "filepath_key": "original_input_image",
            "file_suffix": "",
            "z_index": 5,
            "coverage_name": "WS10M",
            "geoserver_style": {
                "regression": [
                    {"opacity": 1, "quantity": "0", "color": "#000dff", "label": "Min"},
                    {"opacity": 1, "quantity": "10", "color": "#05f224", "label": "10"},
                    {"opacity": 1, "quantity": "20", "color": "#ff9100", "label": "20"},
                    {"opacity": 1, "quantity": "30", "color": "#ff1500", "label": "30"},
                    {
                        "opacity": 1,
                        "quantity": "35",
                        "color": "#ff00d9",
                        "label": "Max",
                    },
                ]
            },
        }
    ],
    "model_display_name": "add-layer-sandbox-model",
    "description": "Texas WS",
    "location": "Texas, US",
    "demo": {"demo": True, "section_name": "My Examples"},
}
