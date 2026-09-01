# Generic Python Processors and Inference Guide

This guide provides comprehensive documentation on how to create, manage, and use generic Python processors for custom data processing, as well as how to run inference operations in GEOStudio Core.

## Table of Contents

- [Overview](#overview)
- [Generic Python Processors](#generic-python-processors)
  - [What are Generic Processors?](#what-are-generic-processors)
  - [Creating a Generic Processor](#creating-a-generic-processor)
  - [Retrieving a Generic Processor](#retrieving-a-generic-processor)
  - [Listing Generic Processors](#listing-generic-processors)
  - [Deleting a Generic Processor](#deleting-a-generic-processor)
- [Running Inference](#running-inference)
  - [Prerequisites](#prerequisites)
  - [Creating an Inference Job](#creating-an-inference-job)
  - [Using Generic Processors in Inference](#using-generic-processors-in-inference)
  - [Monitoring Inference Status](#monitoring-inference-status)
- [Complete Workflow Example](#complete-workflow-example)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

GEOStudio Core provides a flexible system for running custom Python processing logic through **Generic Processors** and executing ML model predictions through **Inference Jobs**. This allows you to:

- Upload custom Python scripts for data processing
- Integrate custom processors into inference pipelines
- Run geospatial ML model predictions on satellite imagery
- Monitor and manage inference jobs

---

## Generic Python Processors

### What are Generic Processors?

Generic Processors are custom Python scripts that can be integrated into the inference pipeline to perform specialized data processing tasks. They are executed as part of the inference workflow, typically before pushing results to GeoServer.

**Use Cases:**
- Custom post-processing of model predictions
- Data transformation and filtering
- Cloud masking with custom thresholds
- Integration with external services
- Custom validation logic

### Creating a Generic Processor

Upload a Python file along with metadata to create a generic processor.

#### API Endpoint

```http
POST {base-api-endpoint}/v2/generic-processor
```

#### Request Format

This endpoint uses `multipart/form-data` to accept both the processor metadata and the Python file.

**Form Fields:**
- `generic_processor` (string, required): JSON string containing processor metadata
- `generic_processor_file` (file, required): Python file (.py extension)

**Metadata Schema:**
```json
{
  "name": "string",
  "description": "string (optional)",
  "processor_parameters": {
    "key": "value"
  }
}
```

#### Example: cURL

```bash

  curl -k -X POST 'https://your-api-url/v2/generic-processor' \
  --header 'accept: application/json' \
  --header "X-API-Key: $STUDIO_API_KEY" \
  --header 'Content-Type: multipart/form-data' \
  -F 'generic_processor_file=@/path/to/your_processor.py;type=text/x-python-script' \
  -F 'generic_processor_metadata={"name":"cloud_masking","description":"Custom cloud masking processor","processor_parameters":{"threshold":80}}'
```

#### Example: Python

```python
import requests
import json
import os

STUDIO_API_KEY = os.getenv("STUDIO_API_KEY", "your-api-key")
file_path = "/path/to/your/script.py"


url = "https://your-api-url/v2/generic-processor"
headers = { "X-API-Key": STUDIO_API_KEY}

# Processor metadata
metadata = {
    "name": "cloud_masking",
    "description": "Custom cloud masking processor",
    "processor_parameters": {"threshold": 80, "method": "scl_based"},
}

# Prepare the multipart form data
files = {"generic_processor_file": ("cloud_masking.py", open(file_path, "rb"))}
data = {"generic_processor_metadata": json.dumps(metadata)}

try:
    response = requests.post(
        url,
        headers=headers,
        files=files,
        data=data,
        verify=False
    )
    processor = response.json()
    if response.status_code == 201:
        print("Succesfully created generic processor")
        print(f"Processor ID: {processor['id']}")   
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

```

#### Response

```json
{
    "name": "cloud_masking",
    "description": "Custom cloud masking processor",
    "processor_parameters": {
        "threshold": 80,
        "method": "scl_based",
    },
    "id": "d5811671-fedb-4863-a711-ede1564ccb65",
    "active": True,
    "created_by": "test@example.com",
    "created_at": "2026-02-10T12:29:42.354388Z",
    "updated_at": "2026-02-10T12:29:42.376796Z",
    "status": "FINISHED",
    "processor_file_path": "d5811671-fedb-4863-a711-ede1564ccb65/cloud_masking.py",
    "processor_presigned_url": None,
}
```

#### Python Processor Template

Your Python processor should be a standalone script that will be executed by the pipeline as `python script.py`. 

This script is expected to have a `__main__` entrypoint. 

The script is also expected to accept an `--input` and `--output` argument that has the path to the input image and the output folder to save the output image.

 The pipeline will pass arguments and parameters to your script.

```python
"""
Generic Python Processor for Custom Data Processing

This script will be executed by the inference pipeline with command-line arguments.
The pipeline passes input, output, and processor parameters.
"""

import sys
import json
import numpy as np
import rasterio
from pathlib import Path
from typing import Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_processing(data: np.ndarray, threshold: int, method: str = 'default') -> np.ndarray:
    """
    Apply your custom processing logic here.
    
    Args:
        data: Input raster data as numpy array
        threshold: Threshold value for processing
        method: Processing method to use
        
    Returns:
        Processed raster data
    """
    logger.info(f"Applying {method} processing with threshold {threshold}")
    
    if method == 'threshold':
        # Apply threshold to mask values
        mask = data > threshold
        processed = np.where(mask, data, 0)
    elif method == 'normalize':
        # Normalize data
        processed = (data - data.min()) / (data.max() - data.min())
    elif method == 'cloud_mask':
        # Example: Cloud masking logic
        # Assuming band 0 is the cloud probability
        cloud_mask = data[0] < threshold
        processed = data.copy()
        processed[:, ~cloud_mask] = 0
    else:
        # Default: return data as-is
        processed = data
    
    return processed


def main(input: str, output: str, parameters: Dict[str, Any]):
    """
    Main processing function.
    
    Args:
        input: Path to input raster file (model output)
        output: Path where processed output should be saved
        parameters: Dictionary of processor parameters from the API
    """
    try:
        logger.info(f"Processing {input}")
        logger.info(f"Parameters: {parameters}")
        
        # Get parameters
        threshold = parameters.get('threshold', 80)
        method = parameters.get('method', 'default')
        
        # Read input raster
        with rasterio.open(input_path) as src:
            data = src.read()
            profile = src.profile
            metadata = src.meta
        
        logger.info(f"Input shape: {data.shape}")
        
        # Apply custom processing logic
        processed_data = apply_processing(data, threshold, method)
        
        logger.info(f"Output shape: {processed_data.shape}")
        
        # Write output
        with rasterio.open(output, 'w', **profile) as dst:
            dst.write(processed_data)
        
        logger.info(f"Successfully processed and saved to {output}")
        
        # Return success status
        result = {
            "status": "success",
            "message": f"Processing completed with threshold {threshold}",
            "output": output,
            "input_shape": list(data.shape),
            "output_shape": list(processed_data.shape)
        }
        
        print(json.dumps(result))
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        
        # Return error status
        result = {
            "status": "error",
            "message": str(e),
            "input_path": input_path
        }
        
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    """
    Entry point when script is executed.
    The pipeline will call this script with arguments:
    python script.py --input <input> --output <output> <your_specific_parameters_values>
    """
    
    if len(sys.argv) < 4:
        logger.error("Usage: python script.py --input <input> --output <output> <your_specific_parameters_values>")
        sys.exit(1)
    
    input = sys.argv[1]
    output = sys.argv[2]
    parameters_json = sys.argv[3]
    
    # Parse parameters
    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse parameters JSON: {e}")
        parameters = {}
    
    # Run main processing
    exit_code = main(input, output, parameters)
    sys.exit(exit_code)
```

**Key Points:**
- The processor is a standalone Python script executed as `python script.py`
- The script receives arbitrary arguments via command line: e.g `input`, `output`, `threshold` ... 
- Include `if __name__ == "__main__":` block as the entry point
- Parse parameters from JSON string passed as argument
- Use proper error handling and logging
- Print results as JSON to stdout for the pipeline to capture
- Return appropriate exit codes (0 for success, 1 for error)

---

### Retrieving a Generic Processor

Get details and download URL for a specific generic processor.

#### API Endpoint

```http
GET /v2/generic-processor/{generic_processor_id}
```

#### Example

```bash
curl -k -X GET "https://your-api-url/v2/generic-processor/processor-id" \
  --header "X-API-Key: $STUDIO_API_KEY" 
```

#### Response

```json
{
  "id": "b1c40d04-1d36-43ed-b733-5dc18aa45689",
  "name": "cloud_masking",
  "description": "Custom cloud masking processor",
  "processor_parameters": {
    "threshold": 80
  },
  "processor_file_path": "b1c40d04-1d36-43ed-b733-5dc18aa45689/cloud_masking.py",
  "processor_presigned_url": "https://s3.us-east-object-storage.appdomain.com/bucket/path?signature=...",
  "status": "FINISHED",
  "created_at": "2026-01-21T10:30:00Z",
  "updated_at": "2026-01-21T10:30:00Z"
}
```

**Note:** The `processor_presigned_url` is valid for 8 hours (28800 seconds).

---

### Listing Generic Processors

Retrieve all generic processors you have access to.

#### API Endpoint

```http
GET /v2/generic-processor
```

#### Example

```bash
curl -k -X GET "https://your-api-url/v2/generic-processor" \
  --header "X-API-Key: $STUDIO_API_KEY" \
```

#### Response

```json
{
  "results": [
    {
      "id": "b1c40d04-1d36-43ed-b733-5dc18aa45689",
      "name": "cloud_masking",
      "description": "Custom cloud masking processor",
      "processor_parameters": {"threshold": 80},
      "status": "FINISHED",
      "created_at": "2026-01-21T10:30:00Z"
    },
    {
      "id": "f315666e-39c9-4144-9175-7964ea208d25",
      "name": "water_detection",
      "description": "Custom water body detection",
      "processor_parameters": {},
      "status": "FINISHED",
      "created_at": "2026-01-20T15:20:00Z"
    }
  ],
  "total_records": 2
}
```

---

### Deleting a Generic Processor

Soft delete a generic processor (marks as inactive).

#### API Endpoint

```http
DELETE /v2/generic-processor/{generic_processor_id}
```

#### Example

```bash
curl -X DELETE "https://your-api-url/v2/generic-processor/b1c40d04-1d36-43ed-b733-5dc18aa45689" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Response

```
HTTP 204 No Content
```

---

## Running Inference

### Prerequisites

Before running inference, ensure you have:

1. **A deployed model** in `FINISHED` or `READY` status
2. **Model metadata** including `model_input_data_spec` and `geoserver_push` configuration
3. **Spatial domain** (bounding boxes, polygons, tiles, or URLs)
4. **Temporal domain** (date range)
5. **(Optional) Generic processor** for custom processing

---

### Creating an Inference Job 

Run ML model predictions on geospatial data.

#### API Endpoint

```http
POST /v2/inference
```

#### Request Schema

```json
{
  "model_display_name": "geofm-sandbox-models",
  "name": "test_generic",
  "description": "testing inference with generic processor",
  "location": "vienna",
  "spatial_domain": {
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]
  },
  "temporal_domain": ["2026-01-21"],
  "fine_tuning_id": "your_uploaded_tune_id",
  "post_processing": {
        "cloud_masking": "False",
        "ocean_masking": "False",
        "snow_ice_masking": null,
        "permanent_water_masking": "False"
    },
}
```

#### Required Fields

- **Either** `model_id` **or** `model_display_name`
- `spatial_domain`: At least one of `bbox`, `polygons`, `tiles`, or `urls`
- `temporal_domain`: Array of date strings

#### Example: Basic Inference

```bash
curl -k -X 'POST' \
  'https://your-api-url/v2/inference' \
  -H 'accept: application/json' \
  -H "X-API-Key: $STUDIO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
  "model_display_name": "geofm-sandbox-models",
  "name": "test_generic",
  "description": "testing inference with generic processor",
  "location": "vienna",
  "spatial_domain": {
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]
  },
  "temporal_domain": ["2026-01-21"],
  "fine_tuning_id": "your_uploaded_tune_id",
  "post_processing": {
        "cloud_masking": "False",
        "ocean_masking": "False",
        "snow_ice_masking": null,
        "permanent_water_masking": "False"
    },
}'
```

#### Example: Python

```python
import requests
import json
import os

STUDIO_API_KEY = os.getenv("STUDIO_API_KEY", "your-api-key")


url = "https://your-api-url/v2/inference"
headers = { "X-API-Key": STUDIO_API_KEY}

payload = {
    "model_display_name": "geofm-sandbox-models",
    "description": "Flood detection for Vienna region",
    "location": "Vienna, Austria",
    "spatial_domain": {
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]},
    },
    "temporal_domain": ["2024-06-15", "2024-06-16"],
    "maxcc": 80,
    "post_processing": {
        "cloud_masking": "False",
        "ocean_masking": "False",
        "snow_ice_masking": "False",
        "permanent_water_masking": "False"
    },
    "fine_tuning_id": "your_uploaded_tune_id"
}

try:
    response = requests.post(url, headers=headers, json=payload, verify=False)
    inference = response.json()

    if response.status_code == 201:
        print("Succesfully created inference")
        print(f"Inference ID: {inference['id']}")
        print(inference)
    else:
        print(f"Error {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

```

#### Response

```json
{
    "spatial_domain": {
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]},
        "polygons": [],
        "tiles": [],
        "urls": []
    },
    "temporal_domain": ["2024-06-15","2024-06-16"],
    "fine_tuning_id": "geotune-gr7oqzxdm6gqk87oswh4xw",
    "generic_processor": None,
    "maxcc": 80,
    "model_display_name": "geofm-sandbox-models",
    "description": "Flood detection for Vienna region",
    "location": "Vienna, Austria",
    "geoserver_layers": None,
    "demo": None,
    "model_id": "3d5828b4-8884-40cb-b67c-bb070e73fe39",
    "inference_output": None,
    "generic_processor_id": None,
    "id": "0aa259c4-c33e-459d-88ce-c4e035de5be2",
    "active": True,
    "created_by": "test@example.com",
    "created_at": "2026-02-11T13: 56: 31.782816Z",
    "updated_at": "2026-02-11T13: 56: 31.789484Z",
    "status": "PENDING",
    "tasks_count_total": 1,
    "tasks_count_success": 0,
    "tasks_count_failed": 0,
    "tasks_count_stopped": 0,
    "tasks_count_waiting": 1
}
```

---

### Using Generic Processors in Inference

Integrate a custom Python processor into your inference pipeline by including the `generic_processor_id`.

#### Example

```bash
curl -k -X 'POST' \
  'https://your-api-url/v2/inference' \
  -H 'accept: application/json' \
  -H "X-API-Key: $STUDIO_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{
  "model_display_name": "geofm-sandbox-models",
  "name": "test_generic",
  "description": "testing inference with generic processor",
  "location": "vienna",
  "spatial_domain": {
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]
  },
  "temporal_domain": ["2026-01-21"],
  "fine_tuning_id": "your_uploaded_tune_id",
  "post_processing": {
        "cloud_masking": "False",
        "ocean_masking": "False",
        "snow_ice_masking": null,
        "permanent_water_masking": "False"
    },
"generic_processor_id": "your_created_generic_id"
}'
```

#### How It Works

1. The generic processor is automatically inserted into the pipeline before the `push-to-geoserver` step
2. The processor receives the model's output as input
3. It applies custom processing logic using the parameters defined during creation
4. The processed output is then pushed to GeoServer

#### Pipeline Steps with Generic Processor

```json
{
  "pipeline_steps": [
    {
      "status": "WAITING",
      "process_id": "url-connector",
      "step_number": 0
    },
    {
      "status": "WAITING",
      "process_id": "terratorch-inference",
      "step_number": 1
    },
    {
      "status": "WAITING",
      "process_id": "postprocess-generic",
      "step_number": 2
    },
    {
      "status": "WAITING",
      "process_id": "generic-python-processor",
      "step_number": 3
    },
    {
      "status": "WAITING",
      "process_id": "push-to-geoserver",
      "step_number": 4
    }
  ]
}
```

---

### Monitoring Inference Status

#### Get Inference Details

```http
GET v2/inference/{inference_id}
```

**Example:**

```bash
curl -k -X GET "https://your-api-url/v2/inference/904d1e13-ddd2-415f-a963-120d16a240f0" \
  --header "X-API-Key: $STUDIO_API_KEY" 
```

**Response:**

```json
{
  "id": "904d1e13-ddd2-415f-a963-120d16a240f0",
  "status": "RUNNING",
  "tasks_count_total": 4,
  "tasks_count_success": 2,
  "tasks_count_failed": 0,
  "tasks_count_stopped": 0,
  "tasks_count_waiting": 2,
  "updated_at": "2026-01-21T10:35:00Z"
}
```

#### Status Values

- `PENDING`: Job created, waiting to start
- `READY`: Job ready to be picked by the pipelines
- `RUNNING`: Currently executing
- `COMPLETED`: All tasks completed successfully
- `FAILED`: One or more tasks failed
- `STOPPED`: Manually stopped

#### Get Inference Tasks

```http
GET /v2/inference/{inference_id}/tasks
```

**Example:**

```bash
curl -X GET "https://your-api-url/v2/inference/904d1e13-ddd2-415f-a963-120d16a240f0/tasks" \
  --header "X-API-Key: $STUDIO_API_KEY" 
```

#### Get Task Output URL

```http
GET /v2/tasks/{task_id}/output-url
```

**Example:**

```bash
curl -X GET "https://your-api-url/v2/tasks/904d1e13-ddd2-415f-a963-120d16a240f0-tile-001/output-url" \
  --header "X-API-Key: $STUDIO_API_KEY" 
```

---

## Complete Workflow Example

This example demonstrates the complete workflow from creating a generic processor to running inference.

```python
import requests
import json
import os
import time

STUDIO_API_KEY = os.getenv("STUDIO_API_KEY", "your-api-key")

# Step 1: Create a generic processor

file_path = "/path/to/your/script.py"

url = "https://your-api-url/v2/"

headers = { "X-API-Key": STUDIO_API_KEY}

# Processor metadata
metadata = {
    "name": "cloud_masking",
    "description": "Custom cloud masking processor",
    "processor_parameters": {"threshold": 80, "method": "scl_based"},
}

# Prepare the multipart form data
files = {"generic_processor_file": ("cloud_masking.py", open(file_path, "rb"))}
data = {"generic_processor_metadata": json.dumps(metadata)}

try:
    response = requests.post(
        f"{url}/generic-processor",
        headers=headers,
        files=files,
        data=data,
        verify=False
    )
    processor = response.json()
    if response.status_code == 201:
        print("Succesfully created generic processor")
        print(f"Processor ID: {processor['id']}")   
    else:
        print(f"Error: {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

# Step 2: Create inference with the generic processor
payload = {
    "model_display_name": "geofm-sandbox-models",
    "description": "Flood detection for Vienna region",
    "location": "Vienna, Austria",
    "spatial_domain": {        
        "urls": [
            "https://geospatial-studio-example-data.s3.us-east.cloud-object-storage.appdomain.cloud/examples-for-inference/austin1_tile_0_1024_train.tiff"
        ]},
    "temporal_domain": ["2024-06-15", "2024-06-16"],
    "maxcc": 80,
        "post_processing": {
        "cloud_masking": "False",
        "ocean_masking": "False",
        "snow_ice_masking": "False",
        "permanent_water_masking": "False"
    },
    "fine_tuning_id": "your_onboarded_tune_id",
    "generic_processor_id": processor['id']

}

try:
    response = requests.post(f"{url}/inference", headers=headers, json=payload, verify=False)
    inference = response.json()

    if response.status_code == 201:
        print("Succesfully created inference")
        inference_id = inference['id']
        print(f"Inference ID: {inference_id}")
        print(inference)
    else:
        print(f"Error {response.status_code} - {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")

# Step 3: Monitor inference progress
while True:
    response = requests.get(
        f"{url}/inference/{inference_id}",
        headers=headers,
        verify=False
    )
    inference = response.json()
    status = inference['status']
    
    print(f"  Status: {status}")
    print(f"  Tasks - Total: {inference.get('tasks_count_total', 0)}, "
          f"Success: {inference.get('tasks_count_success', 0)}")
    
    if status in ['COMPLETED', 'FAILED', 'STOPPED']:
        break
    
    time.sleep(30)  # Check every 30 seconds

# Step 4: Retrieve results
if status == 'COMPLETED':
    print("\n✓ Inference completed successfully!")
    
    response = requests.get(
        f"{url}/inference/{inference_id}/tasks",
        headers=headers,
        verify=False
    )
    tasks = response.json()
    
    for task in tasks['tasks']:
        if task['status'] == 'FINISHED':
            task_id = task['task_id']
            response = requests.get(
                f"{url}/tasks/{task_id}/output-url",
                headers=headers
            )
            if response.status_code == 200:
                output_data = response.json()
                print(f"  Output: {output_data['url']}")
else:
    print(f"\n✗ Inference {status.lower()}")
```

---

## Best Practices

### Creating Generic Processors

1. **Error Handling**: Include comprehensive error handling
2. **Logging**: Use proper logging for debugging
3. **Parameters**: Make processors configurable
4. **Testing**: Test locally before uploading
5. **Documentation**: Include docstrings and comments
6. **File Size**: Keep under 10MB for optimal performance

<!-- ### Inference Jobs

1. **Spatial Domain**: Use appropriate bounding box sizes
2. **Temporal Domain**: Limit date ranges
3. **Cloud Coverage**: Set appropriate `maxcc` values
4. **Monitoring**: Check status for long-running jobs
5. **Resource Management**: Delete completed inferences when done -->

---

## Troubleshooting

### Generic Processor Upload Fails

**Problem**: Processor creation returns 500 error

**Solutions**:
- Download logs from the gateway and see what the issue is. Could be:
  - File doesn't have `.py` extension
  - Validate metadata JSON
  - API key permissions
  - Too large file size. (Timeout)
  - Your code has no way to accept the --input and --output parameters.


### Generic Processor Not Executing

**Problem**: Processor step is skipped or fails

**Solutions**:
- Verify processor ID is correct
- Check processor status is FINISHED
- Ensure processor file is valid Python file
- Review processor logs for errors
- Ensure that the python file has an entrypoint `__main__`
- Add an --input and --output parameter to the python file

---

## API Reference Summary

### Generic Processor Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `v2/generic-processor` | Create processor |
| GET | `v2/generic-processor/{id}` | Get processor |
| GET | `v2/generic-processor` | List processors |
| DELETE | `v2/generic-processor/{id}` | Delete processor |

### Inference Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `v2/inference` | Create inference |
| GET | `v2/inference/{id}` | Get inference |
| GET | `v2/inference` | List inferences |
| GET | `v2/inference/{id}/tasks` | Get tasks |
| DELETE | `v2/inference/{id}` | Delete inference |

### Task Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `v2/tasks/{task_id}/output-url` | Get output URL |
| GET | `v2/tasks/{task_id}/logs/{step_id}` | Get logs |

---

## Related Documentation

- [GEOStudio Core README](./README.md)
- [API Documentation]([https://your-api-url/docs](https://terrastackai.github.io/geospatial-studio-toolkit/api/))
- [Contributing Guide](./CONTRIBUTING.md)

---

## Support

For help:
- Open an issue on [GitHub](https://github.com/terrastackai/geospatial-studio-core/issues)
- Review [Contributing Guide](./CONTRIBUTING.md)
- Check [Code of Conduct](./CODE_OF_CONDUCT.md)
