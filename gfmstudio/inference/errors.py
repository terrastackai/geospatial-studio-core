# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# exceptions.py and having our base class and have our subclasses
# git submodule with exceptions
# close interaction with classes
# list of error codes that we agree on (downstream tasks )

errors = {
    "1001": {
        "downstreamMessage": "No data provided/available for the location and time of interest",
        "uiMessage": "No data available for location/date.",
        "downstreamModule": "dataService",
    },
    "1002": {
        "downstreamMessage": "Input image is too small. Both dimensions must be >= 224",
        "uiMessage": "Image smaller than 224 x 224.",
        "downstreamModule": "dataService",
    },
    "1003": {
        "downstreamMessage": "Invalid eis_data_type of",
        "uiMessage": "Invalid data type.",
        "downstreamModule": "dataService",
    },
    "1004": {
        "downstreamMessage": "COS bucket upload error",
        "uiMessage": "Invalid output_url.",
        "downstreamModule": "dataService",
    },
    "1005": {
        "downstreamMessage": "Event_id/UUID is already in use",
        "uiMessage": "Event_id/UUID is already in use.",
        "downstreamModule": "dataService",
    },
    "1006": {
        "downstreamMessage": "Unable to get data from pre-signed url. Check authentication and expiration.",
        "uiMessage": "Unable to get data from pre-signed url. Check authentication and expiration.",
        "downstreamModule": "dataService",
    },
    "1007": {
        "downstreamMessage": "Input is not a GeoTiff.",
        "uiMessage": "Input is not a GeoTiff.",
        "downstreamModule": "dataService",
    },
    "1008": {
        "downstreamMessage": "EIS data service sdk temporarily unavailable with error.",
        "uiMessage": "EIS data service sdk temporarily unavailable with error.",
        "downstreamModule": "dataService",
    },
    "1009": {
        "downstreamMessage": "Error getting data",
        "uiMessage": "Error getting data.",
        "downstreamModule": "dataService",
    },
    "1010": {
        "downstreamMessage": "Issue with EIS dataservice query",
        "uiMessage": "Issue with EIS dataservice query.",
        "downstreamModule": "dataService",
    },
    "1011": {
        "downstreamMessage": "Invalid url",
        "uiMessage": "Invalid url.",
        "downstreamModule": "dataService",
    },
    "1012": {
        "downstreamMessage": "Invalid data type from URL. Must be .zip, .tif, or .tiff.",
        "uiMessage": "Invalid data type from URL. Must be .zip, .tif, or .tiff.",
        "downstreamModule": "dataService",
    },
    "1013": {
        "downstreamMessage": "One of input image bands has no valid pixels.",
        "uiMessage": "One of input image bands has no valid pixels.",
        "downstreamModule": "dataService",
    },
    "1014": {
        "downstreamMessage": "Unable to connect to GeoDN.",
        "uiMessage": "Unable to connect to GeoDN.",
        "downstreamModule": "dataService",
    },
    "1015": {
        "downstreamMessage": "Unable to connect to STAC data catalog.",
        "uiMessage": "Unable to connect to STAC data catalog.",
        "downstreamModule": "dataService",
    },
    "1016": {
        "downstreamMessage": "Error when downloading data from GeoDN.",
        "uiMessage": "openEO connection issue.",
        "downstreamModule": "dataService",
    },
    "1017": {
        "downstreamMessage": "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",  # noqa E501
        "uiMessage": "Connection timeout error.",
        "downstreamModule": "dataService, inferenceService",
    },
    "1018": {
        "downstreamMessage": "No valid images are available for preprocessing; inference failed",
        "uiMessage": "No valid images for preprocessing.",
        "downstreamModule": "inferencePipelineService",
    },
    "1020": {
        "downstreamMessage": "model id for amo model deploy is already in use; inference failed",
        "uiMessage": "Model id in use; inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1021": {
        "downstreamMessage": "one or more presigned urls for amo model deploy is expired; inference failed",
        "uiMessage": "Presigned url for model deploy expired; inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1030": {
        "downstreamMessage": "Inference failed",
        "uiMessage": "Inference service temporarily unavailable.",
        "downstreamModule": "inferenceService",
    },
    "1031": {
        "downstreamMessage": "S3FS error with attached COS bucket.",
        "uiMessage": "S3FS error with attached COS bucket.",
        "downstreamModule": "inferenceService",
    },
    "1032": {
        "downstreamMessage": "Inference failed",
        "uiMessage": "An error occured when running inference.",
        "downstreamModule": "inferenceService",
    },
    "1033": {
        "downstreamMessage": "Failed to load inference output for postprocessing, inference failed",
        "uiMessage": "Failed to load inference output for postprocessing... inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1040": {
        "downstreamMessage": "Inference failed while running preprocessing",
        "uiMessage": "Preprocessing error... Inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1041": {
        "downstreamMessage": "Inference failed while deploying model",
        "uiMessage": "Model deploy error... Inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1042": {
        "downstreamMessage": "Inference failed while making inference request",
        "uiMessage": "Making inference error... Inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1043": {
        "downstreamMessage": "Inference failed while running postprocessing",
        "uiMessage": "Postprocessing error...... Inference failed",
        "downstreamModule": "inferencePipelineService",
    },
    "1044": {
        "downstreamMessage": "Inference failed while running planning",
        "uiMessage": "Planning error...... Inference failed",
        "downstreamModule": "inferencePipelineService",
    },
}
