/*
* © Copyright IBM Corporation 2025
* SPDX-License-Identifier: Apache-2.0
*/


INSERT INTO inf_data_source (data_connector,collection_id,"groups",sharable,id,created_at,updated_at,created_by,updated_by,active,deleted,data_connector_config) VALUES
	 ('sentinelhub','s2_l2a',NULL,true,'7f3e1296-e37b-429e-90ba-06bad499811d'::uuid,'2025-05-30 12:24:44.456','2025-05-30 12:24:44.456','system@ibm.com',NULL,true,NULL,'{
  "bands": [
    {
      "band_name": "B01",
      "resolution": "60m",
      "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)"
    },
    {
      "band_name": "B02",
      "resolution": "10m",
      "RGB_band": "B",
      "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)"
    },
    {
      "band_name": "B03",
      "resolution": "10m",
      "RGB_band": "G",
      "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)"
    },
    {
      "band_name": "B04",
      "resolution": "10m",
      "RGB_band": "R",
      "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)"
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
      "description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)"
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
    },
    {
      "band_name": "SCL",
      "resolution": "20m",
      "description": "Scene classification data, based on Sen2Cor processor, codelist"
    },
    {
      "band_name": "SNW",
      "resolution": "20m",
      "description": "Snow probability, based on Sen2Cor processor"
    },
    {
      "band_name": "CLD",
      "resolution": "20m",
      "description": "Cloud probability, based on Sen2Cor processor"
    }
  ],
  "max_query_bbox_dimension": 244,
  "search": {
    "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
    "filter": "eo:cloud_cover < maxcc"
  },
  "connector": "sentinelhub",
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
}'),
	 ('sentinelhub','s2_l1c',NULL,true,'0614a69c-c7c7-4148-aac6-9968c6edae64'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
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
}'),
	 ('sentinelhub','hls_s30',NULL,false,'b5cb0d43-a7f0-421f-b5b6-65c0b34c0d98'::uuid,'2025-06-09 11:41:48.661','2025-06-09 11:43:55.401','Beldine.Moturi@ibm.com','Beldine.Moturi@ibm.com',false,true,NULL),
	 ('sentinelhub','hls_s30',NULL,false,'067674b4-f5d7-4d9b-87f4-61a2d195e306'::uuid,'2025-06-09 11:41:55.432','2025-06-09 11:44:37.892','Beldine.Moturi@ibm.com','Beldine.Moturi@ibm.com',false,true,NULL),
	 ('sentinelhub','hls_s30',NULL,true,'86988db3-499a-483e-9148-78da1f1256af'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
  "bands": [
    {
      "band_name": "CoastalAerosol",
      "resolution": "30m",
      "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)"
    },
    {
      "band_name": "Blue",
      "resolution": "30m",
      "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)",
      "RGB_band": "B"
    },
    {
      "band_name": "Green",
      "resolution": "30m",
      "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)",
      "RGB_band": "G"
    },
    {
      "band_name": "Red",
      "resolution": "30m",
      "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)",
      "RGB_band": "R"
    },
    {
      "band_name": "RedEdge1",
      "resolution": "30m",
      "description": "Vegetation red edge, 704.1 nm (S2A), 703.8 nm (S2B)"
    },
    {
      "band_name": "RedEdge2",
      "resolution": "30m",
      "description": "Vegetation red edge, 740.5 nm (S2A), 739.1 nm (S2B)"
    },
    {
      "band_name": "RedEdge3",
      "resolution": "30m",
      "description": "Vegetation red edge, 782.8 nm (S2A), 779.7 nm (S2B)"
    },
    {
      "band_name": "NIR_Broad",
      "resolution": "30m",
      "description": "NIR, 832.8 nm (S2A), 833.0 nm (S2B)"
    },
    {
      "band_name": "NIR_Narrow",
      "resolution": "30m",
      "description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)"
    },
    {
      "band_name": "WaterVapor",
      "resolution": "30m",
      "description": "Water vapour, 945.1 nm (S2A), 943.2 nm (S2B)"
    },
    {
      "band_name": "Cirrus",
      "resolution": "30m",
      "description": "SWIR – Cirrus, 1373.5 nm (S2A), 1376.9 nm (S2B)"
    },
    {
      "band_name": "SWIR1",
      "resolution": "30m",
      "description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)"
    },
    {
      "band_name": "SWIR2",
      "resolution": "30m",
      "description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)"
    },
    {
      "band_name": "QA",
      "resolution": "30m",
      "description": "Quality Assessment band, used for cloud masking"
    }
  ],
  "max_query_bbox_dimension": 244,
  "search": {
    "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
    "filter": "eo:cloud_cover < maxcc"
  },
  "connector": "sentinelhub",
  "modality_tag": "HLS_S30",
  "resolution_m": 30,
  "cloud_masking": {
    "band": "QA",
    "encoding": "hls_fmask"
  },
  "query_template": "Template(\"\"\" //VERSION=3 function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
  "collection_name": "hls_s30",
  "data_collection": "DataCollection.HARMONIZED_LANDSAT_SENTINEL",
  "request_input_data": {
    "maxcc": 80,
    "mosaicking_order": "MosaickingOrder.LEAST_CC"
  }
}'),
	 ('sentinelhub','dem',NULL,true,'2bcf1c97-b4ff-4e9b-85aa-2d26d7aee8fb'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
  "bands": [
    {
      "band_name": "DEM",
      "resolution": "30m",
      "description": "Digital Elevation Model"
    }
  ],
  "max_query_bbox_dimension": 244,
  "search": {
    "fields": "",
    "filters": ""
  },
  "connector": "sentinelhub",
  "resolution_m": 30,
  "query_template": "Template(\"\"\" //VERSION=3 \\n\nfunction setup() { return { input: [\"DEM\"], output:{ id: \"default\", bands: 1, sampleType: SampleType.FLOAT32 } } }\nfunction evaluatePixel(sample) { return [sample.DEM] } \"\"\")",
  "collection_name": "dem",
  "modality_tag": "DEM",
  "data_collection": "DataCollection.DEM_COPERNICUS_30",
  "request_input_data": {
    "dummy": "here"
  }
}'),
	 ('sentinelhub','hls_l30',NULL,true,'49c32863-dbfd-4faf-abff-abc9915f94d3'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
    "bands": [
        {
            "band_name": "CoastalAerosol",
            "resolution": "30m",
            "description": "Coastal aerosol, 442.7 nm (S2A), 442.3 nm (S2B)"
        },
        {
            "band_name": "Blue",
            "resolution": "30m",
            "description": "Blue, 492.4 nm (S2A), 492.1 nm (S2B)",
            "RGB_band": "B"
        },
        {
            "band_name": "Green",
            "resolution": "30m",
            "description": "Green, 559.8 nm (S2A), 559.0 nm (S2B)",
            "RGB_band": "G"
        },
        {
            "band_name": "Red",
            "resolution": "30m",
            "description": "Red, 664.6 nm (S2A), 665.0 nm (S2B)",
            "RGB_band": "R"
        },
        {
            "band_name": "NIR_Narrow",
            "resolution": "30m",
            "description": "Narrow NIR, 864.7 nm (S2A), 864.0 nm (S2B)"
        },
        {
            "band_name": "SWIR1",
            "resolution": "30m",
            "description": "SWIR, 1613.7 nm (S2A), 1610.4 nm (S2B)"
        },
        {
            "band_name": "SWIR2",
            "resolution": "30m",
            "description": "SWIR, 2202.4 nm (S2A), 2185.7 nm (S2B)"
        },
        {
            "band_name": "Cirrus",
            "resolution": "30m",
            "description": "SWIR – Cirrus, 1373.5 nm (S2A), 1376.9 nm (S2B)"
        },
        {
            "band_name": "ThermalInfrared1",
            "resolution": "30m",
            "description": "Thermal Infrared band 1, 10.9 µm (S2A), 10.8 µm (S2B)"
        },
        {
            "band_name": "ThermalInfrared2",
            "resolution": "30m",
            "description": "Thermal Infrared band 2, 12.0 µm (S2A), 12.0 µm (S2B)"
        },
        {
            "band_name": "QA",
            "resolution": "30m",
            "description": "Quality Assessment band, used for cloud masking"
        }
    ],
    "max_query_bbox_dimension": 244,
    "search": {
        "fields": "{\"include\": [\"id\", \"properties.datetime\", \"properties.eo:cloud_cover\"], \"exclude\": []}",
        "filter": "eo:cloud_cover < maxcc"
    },
    "connector": "sentinelhub",
    "modality_tag": "HLS_L30",
    "resolution_m": 30,
    "cloud_masking": {
        "band": "QA",
        "encoding": "hls_fmask"
    },
  "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"DN\" }], output: { bands: ${num_bands}, sampleType: \"INT16\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
    "collection_name": "hls_l30",
    "data_collection": "DataCollection.HARMONIZED_LANDSAT_SENTINEL",
    "request_input_data": {
        "maxcc": 80,
        "mosaicking_order": "MosaickingOrder.LEAST_CC"
    }
}'),
	 ('sentinelhub','s1_grd',NULL,true,'1914eea0-939d-467b-a23f-49bc82fd7b27'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
  "bands": [
    {
      "band_name": "VV",
      "resolution": "10m",
      "description": "Vertical transmit, vertical receive polarization"
    },
    {
      "band_name": "VH",
      "resolution": "10m",
      "description": "Vertical transmit, horizontal receive polarization"
    },
    {
      "band_name": "HV",
      "resolution": "10m",
      "description": "Horizontal transmit, vertical receive polarization"
    },
    {
      "band_name": "HH",
      "resolution": "10m",
      "description": "Horizontal transmit, horizontal receive polarization"
    }
  ],
  "max_query_bbox_dimension": 244,
  "search": {
    "fields": "{\"include\": [\"id\", \"properties.datetime\"], \"exclude\": []}",
    "filters": ""
  },
  "connector": "sentinelhub",
  "modality_tag": "S1GRD",
  "resolution_m": 10,
  "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"LINEAR_POWER\" }], output: { bands: ${num_bands}, sampleType: \"FLOAT32\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
  "collection_name": "sentinel1_vv-vh",
  "data_collection": "DataCollection.SENTINEL1_IW",
  "request_input_data": {
    "mosaicking_order": "MosaickingOrder.MOST_RECENT"
  }
}'),
	 ('sentinelhub','sentinel1_vv-vh',NULL,true,'379468a1-08cf-4e04-9604-bc5592ae07ea'::uuid,'2025-06-05 08:41:36.956','2025-06-05 08:41:36.956','system@ibm.com',NULL,true,NULL,'{
  "bands": [
    {
      "band_name": "VV",
      "resolution": "10m",
      "description": "Vertical transmit, vertical receive polarization"
    },
    {
      "band_name": "VH",
      "resolution": "10m",
      "description": "Vertical transmit, horizontal receive polarization"
    },
    {
      "band_name": "HV",
      "resolution": "10m",
      "description": "Horizontal transmit, vertical receive polarization"
    },
    {
      "band_name": "HH",
      "resolution": "10m",
      "description": "Horizontal transmit, horizontal receive polarization"
    }
  ],
  "max_query_bbox_dimension": 244,
  "search": {
    "fields": "{\"include\": [\"id\", \"properties.datetime\"], \"exclude\": []}",
    "filters": ""
  },
  "connector": "sentinelhub",
  "modality_tag": "S1GRD",
  "resolution_m": 10,
  "query_template": "Template(\"\"\" //VERSION=3 \\n function setup() { return { input: [{ bands: ${bands}, units: \"LINEAR_POWER\" }], output: { bands: ${num_bands}, sampleType: \"FLOAT32\" } }; } function evaluatePixel(sample) { return ${band_samples}; } \"\"\")",
  "collection_name": "sentinel1_vv-vh",
  "data_collection": "DataCollection.SENTINEL1_IW",
  "request_input_data": {
    "mosaicking_order": "MosaickingOrder.MOST_RECENT"
  }
}');
