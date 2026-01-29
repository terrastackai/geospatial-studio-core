# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import uuid
from unittest.mock import patch


class ResourceTestSuiteBase:
    data: list
    endpoint: str

    def setup_method(self):
        self.data = self.data or []
        self.endpoint = self.endpoint or ""

    def test_list_resources__success(self, client):
        response = client.get(self.endpoint)
        assert response.status_code == 200
        assert isinstance(response.json()["results"], list)

    def test_retrieve_missing_record(self, client, db):
        response = client.get(f"{self.endpoint}{str(uuid.uuid4())}")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_create_retrieve_resource__success(self, client, db):
        # Create a new resource
        response = client.post(self.endpoint, json=self.data)
        assert response.status_code == 201

        # Retrieve the created resource
        response_data = response.json()
        resource_id = response_data["id"]
        item_response = client.get(f"{self.endpoint}{resource_id}")
        assert item_response.status_code == 200
        assert item_response.json()["id"] == resource_id

    def test_delete_resource__success(self, client, db):
        # Create a new resource
        response = client.post(self.endpoint, json=self.data)
        assert response.status_code == 201

        # Delete the created resource
        response_data = response.json()
        resource_id = response_data["id"]
        item_response = client.delete(f"{self.endpoint}{resource_id}")
        assert item_response.status_code == 204

        # Verify the resource is deleted
        item_response = client.get(f"{self.endpoint}{resource_id}")
        assert item_response.status_code == 404


class ModelUseCaseSuite(ResourceTestSuiteBase):
    def setup_method(self):
        self.data = {
            "display_name": "flood",
            "description": "Description of flood use case",
            "model_url": "https://amo-prithvi-eo-flood.ibm.com",
        }
        self.endpoint = "/v2/models/"
        super().setup_method()

    def test_advanced_create__success(self, client, db):
        # Create a new resource
        data = {
            "display_name": "AdvancedFloodModel",
            "description": "Advanced model for flood detection",
            "model_url": "https://advanced-flood-model.example.com",
            "pipeline_steps": [
                {
                    "status": "READY",
                    "process_id": "sentinelhub_connector",
                    "step_number": 0,
                },
                {"status": "WAITING", "process_id": "run-inference", "step_number": 1},
                {
                    "status": "WAITING",
                    "process_id": "postprocess-generic",
                    "step_number": 2,
                },
                {
                    "status": "WAITING",
                    "process_id": "push-to-geoserver",
                    "step_number": 3,
                },
            ],
            "geoserver_push": [
                {
                    "workspace": "geofm",
                    "layer_name": "input_rgb",
                    "file_suffix": "",
                    "display_name": "Input image (RGB)",
                    "filepath_key": "model_input_original_image_rgb",
                    "geoserver_style": "",
                },
                {
                    "workspace": "geofm",
                    "layer_name": "pred",
                    "file_suffix": "",
                    "display_name": "Model prediction",
                    "filepath_key": "model_output_image_masked",
                    "geoserver_style": "",
                },
            ],
            "model_input_data_spec": [
                {
                    "bands": {
                        "0": "B02",
                        "1": "B03",
                        "2": "B04",
                        "3": "B8A",
                        "4": "B11",
                        "5": "B12",
                        "6": "SCL",
                    },
                    "connector": "sentinelhub",
                    "collection": "s2_l2a",
                    "scaling_factor": [1, 1, 1, 1, 1, 1, 1],
                }
            ],
            "postprocessing_options": {
                "cloud_masking": "True",
                "ocean_masking": "True",
            },
            "sharable": True,
        }
        response = client.post(self.endpoint, json=data)
        assert response.status_code == 201

        # Retrieve the created resource
        response_data = response.json()
        resource_id = response_data["id"]
        item_response = client.get(f"{self.endpoint}{resource_id}")
        assert item_response.status_code == 200
        assert item_response.json()["id"] == resource_id


class InferenceUseCaseSuite(ResourceTestSuiteBase):
    def setup_method(self):
        self.data = {
            "description": "LA wildfire test",
            "location": "Test Location",
            "model_display_name": "prithvi-eo-fire-scars",
            "spatial_domain": {
                "bbox": [[-119.252472, 33.628342, -117.03650, 34.059309]],
                "polygons": [],
                "tiles": [],
                "urls": [],
            },
            "temporal_domain": [
                "2024-12-18",
                "2025-01-14_2025-01-15",
                "2025-01-20",
            ],
            "data_connector_config": [
                {
                    "connector": "sentinelhub",
                    "collection": "hls_l30",
                    "bands": [
                        {
                            "index": "0",
                            "band_name": "Blue",
                            "scaling_factor": 1,
                            "RGB_band": "B",
                        },
                        {
                            "index": "1",
                            "band_name": "Green",
                            "scaling_factor": 1,
                            "RGB_band": "G",
                        },
                        {
                            "index": "2",
                            "band_name": "Red",
                            "scaling_factor": 1,
                            "RGB_band": "R",
                        },
                        {"index": "3", "band_name": "NIR_Narrow", "scaling_factor": 1},
                        {"index": "4", "band_name": "SWIR1", "scaling_factor": 1},
                        {"index": "5", "band_name": "SWIR2", "scaling_factor": 1},
                    ],
                    "scaling_factor": [
                        0.0001,
                        0.0001,
                        0.0001,
                        0.0001,
                        0.0001,
                        0.0001,
                    ],
                }
            ],
            "post_processing": {"cloud_masking": "False", "ocean_masking": "True"},
            "geoserver_push": [
                {
                    "workspace": "geofm",
                    "layer_name": "input_rgb",
                    "display_name": "Input image (RGB)",
                    "filepath_key": "model_input_original_image_rgb",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
                {
                    "workspace": "geofm",
                    "layer_name": "pred",
                    "display_name": "Model prediction",
                    "filepath_key": "model_output_image_masked",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
            ],
        }
        self.endpoint = "/v2/inference/"
        super().setup_method()

    def test_create_retrieve_resource__success(self, client, db):
        # Attempt submit with missing model
        response = client.post(self.endpoint, json=self.data)
        assert response.status_code == 404
        assert (
            response.json().get("detail")
            == "Model not found. Ensure it exists before retrying."
        )

        # Create the model first
        model_data = {
            "display_name": "prithvi-eo-fire-scars",
            "description": "Fire scars detection model",
            "model_url": "https://example.com/fire-scar-model",
            "geoserver_push": [
                {
                    "workspace": "geofm",
                    "layer_name": "input_rgb",
                    "display_name": "Input image (RGB)",
                    "filepath_key": "model_input_original_image_rgb",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
                {
                    "workspace": "geofm",
                    "layer_name": "pred",
                    "display_name": "Model prediction",
                    "filepath_key": "model_output_image_masked",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
            ],
            "model_input_data_spec": [
                {
                    "bands": [
                        {
                            "index": "0",
                            "band_name": "Blue",
                            "scaling_factor": 1,
                            "RGB_band": "B",
                        },
                        {
                            "index": "1",
                            "band_name": "Green",
                            "scaling_factor": 1,
                            "RGB_band": "G",
                        },
                        {
                            "index": "2",
                            "band_name": "Red",
                            "scaling_factor": 1,
                            "RGB_band": "R",
                        },
                        {"index": "3", "band_name": "NIR_Narrow", "scaling_factor": 1},
                        {"index": "4", "band_name": "SWIR1", "scaling_factor": 1},
                        {"index": "5", "band_name": "SWIR2", "scaling_factor": 1},
                    ],
                    "connector": "sentinelhub",
                    "collection": "hls_l30",
                    "file_suffix": "_merged.tif",
                    "modality_tag": "HLS_L30",
                }
            ],
        }
        model_response = client.post("/v2/models/", json=model_data)
        assert model_response.status_code == 201

        # Now retry creating the inference
        self.data["model_id"] = model_response.json()["id"]
        response = client.post(self.endpoint, json=self.data)
        assert response.status_code == 201

        # Retrieve the created resource
        response_data = response.json()
        resource_id = response_data["id"]
        item_response = client.get(f"{self.endpoint}{resource_id}")
        assert item_response.status_code == 200
        assert item_response.json()["id"] == resource_id

    def test_delete_resource__success(self, client, db):
        # Create the model first
        model_data = {
            "display_name": "prithvi-eo-fire-scars",
            "description": "Fire scars detection model",
            "model_url": "https://example.com/fire-scar-model",
            "geoserver_push": [
                {
                    "workspace": "geofm",
                    "layer_name": "input_rgb",
                    "display_name": "Input image (RGB)",
                    "filepath_key": "model_input_original_image_rgb",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
                {
                    "workspace": "geofm",
                    "layer_name": "pred",
                    "display_name": "Model prediction",
                    "filepath_key": "model_output_image_masked",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
            ],
            "model_input_data_spec": [
                {
                    "bands": [
                        {
                            "index": "0",
                            "band_name": "Blue",
                            "scaling_factor": 1,
                            "RGB_band": "B",
                        },
                        {
                            "index": "1",
                            "band_name": "Green",
                            "scaling_factor": 1,
                            "RGB_band": "G",
                        },
                        {
                            "index": "2",
                            "band_name": "Red",
                            "scaling_factor": 1,
                            "RGB_band": "R",
                        },
                        {"index": "3", "band_name": "NIR_Narrow", "scaling_factor": 1},
                        {"index": "4", "band_name": "SWIR1", "scaling_factor": 1},
                        {"index": "5", "band_name": "SWIR2", "scaling_factor": 1},
                    ],
                    "connector": "sentinelhub",
                    "collection": "hls_l30",
                    "file_suffix": "_merged.tif",
                    "modality_tag": "HLS_L30",
                }
            ],
        }
        model_response = client.post("/v2/models/", json=model_data)
        assert model_response.status_code == 201

        # Now retry creating the inference
        self.data["model_id"] = model_response.json()["id"]
        super().test_delete_resource__success(client, db)

    @patch("gfmstudio.inference.v2.api.invoke_cancel_inference.delay")
    def test_cancel_inference__success(self, mock_delay, client, db):
        mock_delay.return_value = None
        # 1. Create a model post
        model_data = {
            "display_name": "prithvi-eo-fire-scars",
            "description": "Fire scars detection model",
            "model_url": "https://example.com/fire-scar-model",
            "geoserver_push": [
                {
                    "workspace": "geofm",
                    "layer_name": "input_rgb",
                    "display_name": "Input image (RGB)",
                    "filepath_key": "model_input_original_image_rgb",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
                {
                    "workspace": "geofm",
                    "layer_name": "pred",
                    "display_name": "Model prediction",
                    "filepath_key": "model_output_image_masked",
                    "file_suffix": "",
                    "geoserver_style": "",
                },
            ],
            "model_input_data_spec": [
                {
                    "bands": [
                        {
                            "index": "0",
                            "band_name": "Blue",
                            "scaling_factor": 1,
                            "RGB_band": "B",
                        },
                        {
                            "index": "1",
                            "band_name": "Green",
                            "scaling_factor": 1,
                            "RGB_band": "G",
                        },
                        {
                            "index": "2",
                            "band_name": "Red",
                            "scaling_factor": 1,
                            "RGB_band": "R",
                        },
                        {"index": "3", "band_name": "NIR_Narrow", "scaling_factor": 1},
                        {"index": "4", "band_name": "SWIR1", "scaling_factor": 1},
                        {"index": "5", "band_name": "SWIR2", "scaling_factor": 1},
                    ],
                    "connector": "sentinelhub",
                    "collection": "hls_l30",
                    "file_suffix": "_merged.tif",
                    "modality_tag": "HLS_L30",
                }
            ],
        }
        model_response = client.post("/v2/models/", json=model_data)
        assert model_response.status_code == 201

        # 2. From 1 get the id i.e model-id and add it to inference request payload
        # 3. Create an inference with payload from 2
        self.data["model_id"] = model_response.json()["id"]
        response = client.post(self.endpoint, json=self.data)
        assert response.status_code == 201

        # 4. Assert that inference created in 3 has
        # response.status_code == 201 and response.json()["status"] == "PENDING"
        assert response.json()["status"] == "PENDING"
        inf_id = response.json()["id"]

        # 5. MAKE a request to cancel the inference POST /v2/inference/inf-id/cancel
        response = client.post(f"/v2/inference/{inf_id}/cancel")
        # 6. Assert that response in 5 has response.status_code == 202
        assert response.status_code == 202

        # 7  assert that invoke method was called once
        assert mock_delay.called_once()


class TestTuneUseCaseSuite:
    def setup_method(self):
        self.endpoint = "v2/tunes/"
        self.data = {
            "name": "floods-tune",
            "description": "testing retry backoff changes",
            "dataset_id": "sandbox",
            "base_model_id": " base_model_id",
            "tune_template_id": "str(tune_template_id)",
            "tune_config_url": "http://example.com",
            "tune_checkpoint_url": "http://example.com",
            "model_input_data_spec": [
                {
                    "bands": {
                        "0": "B02",
                        "1": "B03",
                        "2": "B04",
                        "3": "B8A",
                        "4": "B11",
                        "5": "B12",
                    },
                    "connector": "sentinelhub",
                    "collection": "s2_l2a",
                    "scaling_factor": [1, 1, 1, 1, 1, 1],
                }
            ],
            "data_connector_config": [
                {
                    "connector": "string",
                    "collection": "string",
                    "bands": [{"additionalProp1": {}}],
                    "scaling_factor": [0],
                    "additionalProp1": {},
                }
            ],
        }

    @patch("gfmstudio.fine_tuning.api.invoke_tune_upload.delay")
    def test_upload_tune_success(self, mock_delay, client, db):
        mock_delay.return_value = None

        # create a default tune template
        tune_template = {
            "id": "template-default",
            "name": "Default Template",
            "purpose": "Segmentation",
            "content": "",
        }
        response = client.post("/v2/tune-templates/", json=tune_template)
        tune_template_id = response.json()["id"]
        self.data["tune_template_id"] = str(tune_template_id)

        # Retrieve the created resource
        item_response = client.post("/v2/upload-completed-tunes", json=self.data)
        assert item_response.status_code == 201

        assert mock_delay.called_once()
