# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import random


def generate_random_colors(n, seed=42):
    random.seed(seed)
    return [
        f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        for _ in range(n)
    ]


def convert_to_geoserver_sld(
    *,
    label_categories: list[dict],
    task_type: str,
) -> list[dict]:
    sld_structure = [
        {
            "workspace": "geofm",
            "layer_name": "input_rgb",
            "display_name": "Input image (RGB)",
            "filepath_key": "model_input_original_image_rgb",
            "file_suffix": "",
            "z_index": 0,
            "visible_by_default": "True",
            "geoserver_style": {
                "rgb": [
                    {
                        "minValue": 0,
                        "maxValue": 255,
                        "channel": 1,
                        "label": "RedChannel",
                    },
                    {
                        "minValue": 0,
                        "maxValue": 255,
                        "channel": 2,
                        "label": "GreenChannel",
                    },
                    {
                        "minValue": 0,
                        "maxValue": 255,
                        "channel": 3,
                        "label": "BlueChannel",
                    },
                ]
            },
        },
        {
            "workspace": "geofm",
            "layer_name": "pred",
            "display_name": "Model prediction",
            "filepath_key": "model_output_image_masked",
            "file_suffix": "",
            "z_index": 1,
            "visible_by_default": "True",
            "geoserver_style": {task_type: []},
        },
    ]

    label_map = []
    default_colors = generate_random_colors(len(label_categories))
    for i, label in enumerate(label_categories):
        label_map.append(
            {
                "color": label.get("color", default_colors[i % len(default_colors)]),
                "quantity": label["id"],
                "opacity": label.get("opacity", 1),
                "label": label["name"].lower().replace(" ", "-"),
            }
        )

    sld_structure[1]["geoserver_style"][task_type] = label_map
    return sld_structure
