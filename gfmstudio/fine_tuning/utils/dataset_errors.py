# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


error_stages = {
    "0001": "Problem downloading dataset",
    "0002": "Problem finding and sorting images",
    "0003": "Problem verifying image dimensions",
    "0004": "Problem obtaining file stems",
    "0005": "Problem creating file splits",
    "0006": "Problem uploading split files",
    "0007": "Problem calculating training parameters",
    "0008": "COG validation issue",
    "0009": "Problem uploading dataset and labels to COS",
    "0010": "Problem populating onboarding details",
    "0011": "Problem pushing training parameters to COS",
}

known_errors = {
    "0001": {
        "unknown url type": "Invalid dataset_url.",
        "HTTP Error 404: Not Found": "Dataset not found. Please verify if the dataset_url is valid.",
        "dataset_url is a required field": "keep original",
    },
    "0002": {},
    "0003": {
        "same dimension as the other images": "keep original",
    },
    "0004": {"don't match": "keep original"},
    "0005": {
        "The split provide isn't valid": "keep original",
    },
    "0006": {},
    "0007": {
        "is out of bounds for axis": "The custom bands is out of bounds.  Please only enter the bands that are available in the dataset.",  # noqa E501
    },
    "0008": {},
    "0009": {},
    "0010": {},
    "0011": {},
}
