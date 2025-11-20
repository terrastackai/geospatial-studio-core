/*
* © Copyright IBM Corporation 2025
* SPDX-License-Identifier: Apache-2.0
*/


INSERT INTO inf_model
(internal_name, display_name, description, model_url, pipeline_steps, geoserver_push, postprocessing_options, status, "groups", sharable, model_onboarding_config, "version", latest, id, created_at, updated_at, created_by, updated_by, active, deleted, model_input_data_spec)
VALUES('prithvi-eo-flood-blair', 'prithvi-eo-flood', 'senflood11_swin with terratorch', 'https://amo-prithvi-eo-flood-blair-internal-nasageospatial-dev.cash.sl.cloud9.ibm.com/v1/caikit.runtime.GeospatialExtension/GeospatialExtensionService/GeospatialPredict', '[
  {
    "status": "READY",
    "process_id": "terrakit-data-fetch",
    "step_number": 0
  },
  {
    "status": "WAITING",
    "process_id": "run-inference",
    "step_number": 1
  },
  {
    "status": "WAITING",
    "process_id": "postprocess-generic",
    "step_number": 2
  },
  {
    "status": "WAITING",
    "process_id": "push-to-geoserver",
    "step_number": 3
  }
]'::json, '[
  {
    "workspace": "geofm",
    "layer_name": "input_rgb",
    "file_suffix": "",
    "display_name": "Input image (RGB)",
    "filepath_key": "model_input_original_image_rgb",
    "z_index": 0,
    "visible_by_default": "True",
    "geoserver_style": {
      "rgb": [
        {
          "minValue": 0,
          "maxValue": 255,
          "channel": 1,
          "label": "RedChannel"
        },
        {
          "minValue": 0,
          "maxValue": 255,
          "channel": 2,
          "label": "GreenChannel"
        },
        {
          "minValue": 0,
          "maxValue": 255,
          "channel": 3,
          "label": "BlueChannel"
        }
      ]
    }
  },
  {
    "workspace": "geofm",
    "layer_name": "pred",
    "file_suffix": "",
    "display_name": "Model prediction",
    "filepath_key": "model_output_image_masked",
    "z_index": 1,
    "visible_by_default": "True",
    "geoserver_style": {
      "segmentation": [
        {
          "color": "#808080",
          "quantity": "0",
          "opacity": 0,
          "label": "No flood"
        },
        {
          "color": "#FA4D56",
          "quantity": "1",
          "opacity": 1,
          "label": "Flood"
        },
        {
          "color": "#4589FF",
          "quantity": 997,
          "opacity": 1,
          "label": "Permanent water"
        },
        {
          "color": "#FFFAFA",
          "quantity": 998,
          "opacity": 1,
          "label": "Snow/ice"
        },
        {
          "color": "#CCCCCC",
          "quantity": 999,
          "opacity": 1,
          "label": "Clouds"
        }
      ]
    }
  }
]'::json, '{"cloud_masking": "True", "ocean_masking": "True"}'::json, 'COMPLETED', NULL, true, '{}'::jsonb, 1.0, true, '88cc4030-0c4a-497c-90cb-3d7a0e9fd371'::uuid, '2025-05-28 12:27:36.865', '2025-05-30 13:43:27.929', 'system@ibm.com', 'Brian.Ogolla@ibm.com', true, false, '[
  {
    "bands": [
      {
        "index": 0,
        "band_name": "B02",
        "scaling_factor": 1,
        "RGB_band": "R"
      },
      {
        "index": 1,
        "band_name": "B03",
        "scaling_factor": 1,
        "RGB_band": "G"
      },
      {
        "index": 2,
        "band_name": "B04",
        "scaling_factor": 1,
        "RGB_band": "B"
      },
      {
        "index": 3,
        "band_name": "B8A",
        "scaling_factor": 1
      },
      {
        "index": 4,
        "band_name": "B11",
        "scaling_factor": 1
      },
      {
        "index": 5,
        "band_name": "B12",
        "scaling_factor": 1
      },
      {
        "index": 6,
        "band_name": "SCL",
        "scaling_factor": 1
      }
    ],
    "connector": "sentinelhub",
    "collection": "s2_l2a",
    "file_suffix": "S2Hand"
  }
]'::json),
('geofm-sandbox-models-v1-4e75e950','geofm-sandbox-models','Early-access test model made available for demonstration or limited user evaluation. These models may include incomplete features or evolving performance characteristics and are intended for feedback and experimentation before full deployment.',NULL,'[{"status": "READY", "process_id": "terrakit-data-fetch", "step_number": 0}, {"status": "WAITING", "process_id": "terratorch-inference", "step_number": 1}, {"status": "WAITING", "process_id": "postprocess-generic", "step_number": 2}, {"status": "WAITING", "process_id": "push-to-geoserver", "step_number": 3}]','[{"workspace": "geofm", "layer_name": "input_rgb", "file_suffix": "", "display_name": "Input image (RGB)", "filepath_key": "model_input_original_image_rgb", "geoserver_style": ""}, {"workspace": "geofm", "layer_name": "pred", "file_suffix": "", "display_name": "Model prediction", "filepath_key": "model_output_image_masked", "geoserver_style": ""}]','{"cloud_masking": "True", "ocean_masking": "True"}','COMPLETED',NULL,true,'{}',1.0,true,'6eeed629-5206-4c1a-8188-58ad754fd235'::uuid,'2025-06-11 09:55:57.456','2025-06-11 09:55:57.485','system@ibm.com','Brian.Ogolla@ibm.com',true,false,'[]');
