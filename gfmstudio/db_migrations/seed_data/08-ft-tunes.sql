/*
* © Copyright IBM Corporation 2025
* SPDX-License-Identifier: Apache-2.0
*/


INSERT INTO ft_tunes (created_by,dataset_id,mcad_id,latest_chkpt,id,created_at,updated_at,updated_by,active,deleted,name,description,tune_template_id,config,status,progress,metrics,logs,config_json,base_model_id,model_parameters,train_options,tuning_config) VALUES
	 ('system@ibm.com','geodata-test','kjob-geotune-99t33v4padveblaskapgra',NULL,'geotune-99t33v4padveblaskapgra','2025-06-12 10:43:25.873','2025-06-12 10:43:25.873','system@ibm.com',true,false,'granite-geospatial-land-surface-temperature','granite-geospatial-land-surface-temperature','d1137e61-58dc-4c56-b9db-25474e0944ad'::uuid,NULL,'Finished',NULL,NULL,NULL,NULL,NULL,'{}','{}','tune-tasks/geotune-99t33v4padveblaskapgra/config_deploy.yaml'),
	 ('system@ibm.com','geodata-test','kjob-geotune-gvw9bewpunjxtw5x5nvexp',NULL,'geotune-gvw9bewpunjxtw5x5nvexp','2025-06-12 10:44:30.018','2025-06-12 10:44:30.018','system@ibm.com',true,false,'granite-geospatial-uki-flooddetection','granite-geospatial-uki-flooddetection','e4791b2c-bb17-4a5e-9f05-1be5411a4fa6'::uuid,NULL,'Finished',NULL,NULL,NULL,NULL,NULL,'{}','{
  "model_framework": "terratorch-v2",
  "model_input_data_spec": [
    {
      "bands": {
        "0": "B02",
        "1": "B03",
        "2": "B04",
        "3": "B8A",
        "4": "B11",
        "5": "B12",
        "6": "SCL"
      },
      "connector": "sentinelhub",
      "collection": "s2_l2a",
      "scaling_factor": [
        1,
        1,
        1,
        1,
        1,
        1,
        1
      ]
    }
  ],
  "data_connector_config": [
    {
      "bands": {
        "10m": [
          "B02",
          "B03",
          "B04",
          "B08",
          "AOT"
        ],
        "20m": [
          "B05",
          "B06",
          "B07",
          "B8A",
          "B11",
          "B12",
          "SCL",
          "SNW",
          "CLD"
        ],
        "60m": [
          "B01",
          "B09"
        ]
      },
      "search": {
        "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
        "filter": "eo:cloud_cover < maxcc"
      },
      "connector": "sentinelhub",
      "rgb_bands": [
        "B04",
        "B03",
        "B02"
      ],
      "modality_tag": "S2L2A",
      "resolution_m": 10,
      "cloud_masking": {
        "band": "SCL",
        "encoding": "sentinel2_scl"
      },
      "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
      "collection_name": "s2_l2a",
      "data_collection": "DataCollection.SENTINEL2_L2A",
      "request_input_data": {
        "maxcc": 80,
        "mosaicking_order": "MosaickingOrder.LEAST_CC"
      }
    }
  ],
  "geoserver_push": [
    {
      "workspace": "geofm",
      "layer_name": "input_rgb",
      "display_name": "Input image (RGB)",
      "filepath_key": "model_input_original_image_rgb",
      "file_suffix": "",
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
      "display_name": "Model prediction",
      "filepath_key": "model_output_image_masked",
      "file_suffix": "",
      "geoserver_style": {
        "segmentation": [
          {
            "color": "#808080",
            "quantity": "0",
            "opacity": 1,
            "label": "no flood"
          },
          {
            "color": "#ff08be",
            "quantity": "1",
            "opacity": 1,
            "label": "flood"
          },
          {
            "color": "#ff00ee",
            "quantity": 998,
            "opacity": 1,
            "label": "permanent-water"
          }
        ]
      }
    }
  ]
}','tune-tasks/geotune-gvw9bewpunjxtw5x5nvexp/config_deploy.yaml'),
	 ('system@ibm.com','geodata-test','kjob-geotune-sksuxqstfgujhmwmfgjshg',NULL,'geotune-sksuxqstfgujhmwmfgjshg','2025-06-12 10:46:05.031','2025-06-12 10:46:05.031','system@ibm.com',true,false,'granite-geospatial-canopyheight','granite-geospatial-canopyheight','d1137e61-58dc-4c56-b9db-25474e0944ad'::uuid,NULL,'Finished',NULL,NULL,NULL,NULL,NULL,'{}','{
  "model_framework": "terratorch-v2",
  "model_input_data_spec": [
    {
      "bands": {
        "0": "B02",
        "1": "B03",
        "2": "B04",
        "3": "B8A",
        "4": "B11",
        "5": "B12"
      },
      "connector": "sentinelhub",
      "collection": "s2_l2a",
      "scaling_factor": [
        1,
        1,
        1,
        1,
        1,
        1
      ]
    }
  ],
  "data_connector_config": [
    {
      "bands": {
        "10m": [
          "B02",
          "B03",
          "B04",
          "B08",
          "AOT"
        ],
        "20m": [
          "B05",
          "B06",
          "B07",
          "B8A",
          "B11",
          "B12",
          "SCL",
          "SNW",
          "CLD"
        ],
        "60m": [
          "B01",
          "B09"
        ]
      },
      "search": {
        "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
        "filter": "eo:cloud_cover < maxcc"
      },
      "connector": "sentinelhub",
      "rgb_bands": [
        "B04",
        "B03",
        "B02"
      ],
      "modality_tag": "S2L2A",
      "resolution_m": 10,
      "cloud_masking": {
        "band": "SCL",
        "encoding": "sentinel2_scl"
      },
      "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
      "collection_name": "s2_l2a",
      "data_collection": "DataCollection.SENTINEL2_L2A",
      "request_input_data": {
        "maxcc": 80,
        "mosaicking_order": "MosaickingOrder.LEAST_CC"
      }
    }
  ],
  "geoserver_push": [
    {
      "workspace": "geofm",
      "layer_name": "input_rgb",
      "display_name": "Input image (RGB)",
      "filepath_key": "model_input_original_image_rgb",
      "file_suffix": "",
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
      "display_name": "Model prediction",
      "filepath_key": "model_output_image_masked",
      "file_suffix": "",
      "geoserver_style": {
        "segmentation": [
          {
            "color": "#808080",
            "quantity": "0",
            "opacity": 1,
            "label": "no flood"
          },
          {
            "color": "#ff08be",
            "quantity": "1",
            "opacity": 1,
            "label": "flood"
          },
          {
            "color": "#ff00ee",
            "quantity": 998,
            "opacity": 1,
            "label": "permanent-water"
          }
        ]
      }
    }
  ]
}','tune-tasks/geotune-sksuxqstfgujhmwmfgjshg/config_deploy.yaml'),
	 ('system@ibm.com','geodata-test','kjob-geotune-euz7jbdkusejbywykrviza',NULL,'geotune-euz7jbdkusejbywykrviza','2025-06-12 10:38:24.693','2025-06-12 10:38:24.693','system@ibm.com',true,false,'granite-geospatial-biomass','granite-geospatial-biomass','d1137e61-58dc-4c56-b9db-25474e0944ad'::uuid,NULL,'Finished',NULL,NULL,NULL,NULL,NULL,'{}','{
  "model_framework": "terratorch-v2",
  "model_input_data_spec": [
    {
      "bands": {
        "0": "B02",
        "1": "B03",
        "2": "B04",
        "3": "B8A",
        "4": "B11",
        "5": "B12"
      },
      "connector": "sentinelhub",
      "collection": "s2_l2a",
      "scaling_factor": [
        1,
        1,
        1,
        1,
        1,
        1
      ]
    }
  ],
  "data_connector_config": [
    {
      "bands": {
        "10m": [
          "B02",
          "B03",
          "B04",
          "B08",
          "AOT"
        ],
        "20m": [
          "B05",
          "B06",
          "B07",
          "B8A",
          "B11",
          "B12",
          "SCL",
          "SNW",
          "CLD"
        ],
        "60m": [
          "B01",
          "B09"
        ]
      },
      "search": {
        "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
        "filter": "eo:cloud_cover < maxcc"
      },
      "connector": "sentinelhub",
      "rgb_bands": [
        "B04",
        "B03",
        "B02"
      ],
      "modality_tag": "S2L2A",
      "resolution_m": 10,
      "cloud_masking": {
        "band": "SCL",
        "encoding": "sentinel2_scl"
      },
      "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
      "collection_name": "s2_l2a",
      "data_collection": "DataCollection.SENTINEL2_L2A",
      "request_input_data": {
        "maxcc": 80,
        "mosaicking_order": "MosaickingOrder.LEAST_CC"
      }
    }
  ],
  "geoserver_push": [
    {
      "workspace": "geofm",
      "layer_name": "input_rgb",
      "display_name": "Input image (RGB)",
      "filepath_key": "model_input_original_image_rgb",
      "file_suffix": "",
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
      "display_name": "Model prediction",
      "filepath_key": "model_output_image_masked",
      "file_suffix": "",
      "geoserver_style": {
        "segmentation": [
          {
            "color": "#808080",
            "quantity": "0",
            "opacity": 1,
            "label": "no flood"
          },
          {
            "color": "#ff08be",
            "quantity": "1",
            "opacity": 1,
            "label": "flood"
          },
          {
            "color": "#ff00ee",
            "quantity": 998,
            "opacity": 1,
            "label": "permanent-water"
          }
        ]
      }
    }
  ]
}','tune-tasks/geotune-euz7jbdkusejbywykrviza/config_deploy.yaml'),
	 ('system@ibm.com','geodata-test','kjob-geotune-fhxf8ltqw8bddsuojd9ixs',NULL,'geotune-fhxf8ltqw8bddsuojd9ixs','2025-06-17 08:10:13.731','2025-06-17 08:20:41.553','system@ibm.com',true,false,'BrianSegFlood300M','brianseg-0625-0804','e4791b2c-bb17-4a5e-9f05-1be5411a4fa6'::uuid,NULL,'Finished',NULL,'[{"Train": "/experiments/1860/runs/a4b17675b88b4dbcb9f4443c2c169edd"}, {"Test": "/experiments/1860/runs/17d89e02ed2c4bb78dbfb238b3d13825"}]','',NULL,'9b2c4fa9-8478-49d4-9114-b2469d075662'::uuid,'{}','{
  "model_framework": "terratorch-v2",
  "model_input_data_spec": [{
      "bands": [
          {
              "band_name": "B01",
              "index": 0,
              "scaling_factor": 1
          },
          {
              "band_name": "B02",
              "index": 1,
              "scaling_factor": 1,
              "RGB_band": "B"
          },
          {
              "band_name": "B03",
              "index": 2,
              "scaling_factor": 1,
              "RGB_band": "G"
          },
          {
              "band_name": "B04",
              "index": 3,
              "scaling_factor": 1,
              "RGB_band": "R"
          },
          {
              "band_name": "B05",
              "index": 4,
              "scaling_factor": 1
          },
          {
              "band_name": "B06",
              "index": 5,
              "scaling_factor": 1
          },
          {
              "band_name": "B07",
              "index": 6,
              "scaling_factor": 1
          },
          {
              "band_name": "B08",
              "index": 7,
              "scaling_factor": 1
          },
          {
              "band_name": "B8A",
              "index": 8,
              "scaling_factor": 1
          },
          {
              "band_name": "B09",
              "index": 9,
              "scaling_factor": 1
          },
          {
              "band_name": "B10",
              "index": 10,
              "scaling_factor": 1
          },
          {
              "band_name": "B11",
              "index": 11,
              "scaling_factor": 1
          },
          {
              "band_name": "B12",
              "index": 12,
              "scaling_factor": 1
          }
      ],
      "connector": "sentinelhub",
      "collection": "s2_l1c",
      "file_suffix": "S2L1CHand"
  }],  
  "data_connector_config": [{
      "bands": [
          {
              "band_name": "B01",
              "resolution": "60m",
              "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)"
          },
          {
              "band_name": "B02",
              "resolution": "10m",
              "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)",
              "RGB_band": "B"
          },
          {
              "band_name": "B03",
              "resolution": "10m",
              "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)",
              "RGB_band": "G"
          },
          {
              "band_name": "B04",
              "resolution": "10m",
              "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)",
              "RGB_band": "R"
          },
          {
              "band_name": "B05",
              "resolution": "20m",
              "description": "Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)"
          },
          {
              "band_name": "B06",
              "resolution": "20m",
              "description": "Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)"
          },
          {
              "band_name": "B07",
              "resolution": "20m",
              "description": "Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)"
          },
          {
              "band_name": "B08",
              "resolution": "10m",
              "description": "NIR, 832.8 nm (S2A), 833.0 nm (S2B)"
          },
          {
              "band_name": "B8A",
              "resolution": "20m",
              "description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)"
          },
          {
              "band_name": "B09",
              "resolution": "60m",
              "description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B"
          },
          {
              "band_name": "B10",
              "resolution": "60m",
              "description": "SWIR – Cirrus, 1373.5 nm (S2A), 1376.9 nm (S2B)"
          },
          {
              "band_name": "B11",
              "resolution": "20m",
              "description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)"
          },
          {
              "band_name": "B12",
              "resolution": "20m",
              "description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)"
          }
      ],
      "max_query_bbox_dimension": 244,
      "search": {
          "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
          "filter": "eo:cloud_cover < maxcc"
      },
      "cloud_masking": {
          "enabled": true,
          "band": "B08",
          "encoding": "sentinel2_scl"
      },
      "connector": "sentinelhub",
      "modality_tag": "S2L1C",
      "resolution_m": 10,
      "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
      "collection_name": "s2_l1c",
      "data_collection": "DataCollection.SENTINEL2_L1C",
      "request_input_data": {
          "maxcc": 40,
          "mosaicking_order": "MosaickingOrder.LEAST_CC"
      }
  }],
  "geoserver_push": [
      {
          "workspace": "geofm",
          "layer_name": "input_rgb",
          "display_name": "Input image (RGB)",
          "filepath_key": "model_input_original_image_rgb",
          "file_suffix": "",
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
          "display_name": "Model prediction",
          "filepath_key": "model_output_image_masked",
          "file_suffix": "",
          "geoserver_style": {
              "segmentation": [
                  {
                      "color": "#808080",
                      "quantity": "0",
                      "opacity": 1,
                      "label": "no flood"
                  },
                  {
                      "color": "#ff08be",
                      "quantity": "1",
                      "opacity": 1,
                      "label": "flood"
                  },
                  {
                      "color": "#ff00ee",
                      "quantity": 998,
                      "opacity": 1,
                      "label": "permanent-water"
                  }
              ]
          }
      }
  ],
  "post_processing": {
      "cloud_masking": "False",
      "ocean_masking": "True",
      "permanent-water": {
          "enabled": "True",
          "value": 998,
          "color": "#ff00ee"
      }
  }
}','tune-tasks/geotune-fhxf8ltqw8bddsuojd9ixs/geotune-fhxf8ltqw8bddsuojd9ixs_config.yaml'),
	 ('system@ibm.com','geodata-test','kjob-geotune-2qojedwwivmuvpwg2nzkpt',NULL,'geotune-2qojEDwWiVMuvpWG2NZkpT','2024-10-05 00:17:41.039','2024-10-05 01:33:57.487','system@ibm.com',true,false,'flood-segmentation200-mwymqgk6','Segmentation','e4791b2c-bb17-4a5e-9f05-1be5411a4fa6'::uuid,NULL,'Finished',NULL,NULL,'',NULL,'f24fad3d-d5b5-40aa-a8ce-700a1a3d0a83'::uuid,'{"data": {"batch_size": 4}, "model": {"decode_head": {"channels": 32, "num_convs": 1, "loss_decode": {"type": "CrossEntropyLoss", "avg_non_ignore": true}}, "frozen_backbone": false}, "runner": {"max_epochs": "200"}, "lr_config": {"policy": "Fixed"}, "optimizer": {"lr": 6e-05, "type": "Adam"}, "evaluation": {"interval": 2}}','{}',NULL);
