# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
Example payloads for integration tests.
Each payload is a Python dict, instead of JSON files.
"""

# Base valid data-source payload
DATA_SOURCE_SENTINEL = {
    "data_connector": "sentinelhub11",
    "collection_id": "hls_s30",
    "data_source_config": {
        "bands": [
            "CoastalAerosol",
            "Blue",
            "Green",
            "Red",
            "NIR_Narrow",
            "SWIR1",
            "SWIR2",
            "Cirrus",
            "ThermalInfrared1",
            "ThermalInfrared2",
            "QA",
        ],
        "search": {
            "fields": '{"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []}',
            "filter": "eo:cloud_cover < maxcc",
        },
        "connector": "sentinelhub",
        "rgb_bands": ["Red", "Green", "Blue"],
        "resolution_m": 30,
        "cloud_masking": {"band": "QA", "encoding": "hls_fmask"},
        "query_template": 'Template(""" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \\"DN\\" }], output: { bands: ${num_bands}, sampleType: \\"INT16\\" } }; } function evaluatePixel(sample) { return ${band_samples}; } """)',
        "collection_name": "hls_l30",
        "data_collection": "DataCollection.HARMONIZED_LANDSAT_SENTINEL",
        "request_input_data": {
            "maxcc": 80,
            "mosaicking_order": "MosaickingOrder.LEAST_CC",
        },
    },
}
