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
POST {base-api-endpoint}/api/v2/generic-processor
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
curl -X POST "https://your-api-url/api/v2/generic-processor" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F 'generic_processor={"name":"cloud_masking","description":"Custom cloud masking processor","processor_parameters":{"threshold":80}}' \
  -F "generic_processor_file=@/path/to/your_processor.py"
```

#### Example: Python

```python
import requests
import json

url = "https://your-api-url/api/v2/generic-processor"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

# Processor metadata
metadata = {
    "name": "cloud_masking",
    "description": "Custom cloud masking processor",
    "processor_parameters": {
        "threshold": 80,
        "method": "scl_based"
    }
}

# Prepare the multipart form data
files = {
    'generic_processor_file': ('cloud_masking.py', open('cloud_masking.py', 'rb'))
}
data = {
    'generic_processor': json.dumps(metadata)
}

response = requests.post(url, headers=headers, files=files, data=data)
processor = response.json()
print(f"Processor ID: {processor['id']}")
```

#### Response

```json
{
  "id": "b1c40d04-1d36-43ed-b733-5dc18aa45689",
  "name": "cloud_masking",
  "description": "Custom cloud masking processor",
  "processor_parameters": {
    "threshold": 80,
    "method": "scl_based"
  },
  "processor_file_path": "b1c40d04-1d36-43ed-b733-5dc18aa45689/cloud_masking.py",
  "status": "FINISHED",
  "created_at": "2026-01-21T10:30:00Z",
  "updated_at": "2026-01-21T10:30:00Z",
  "created_by": "user@example.com"
}
```

#### Python Processor Template

Your Python processor should be a standalone script that will be executed by the pipeline as `python script.py`. This script is expected to have a `__main__` entrypoint. The pipeline will pass arguments and parameters to your script.

```python
"""
Generic Python Processor for Custom Data Processing

This script will be executed by the inference pipeline with command-line arguments.
The pipeline passes input_path, output_path, and processor parameters.
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


def main(input_path: str, output_path: str, parameters: Dict[str, Any]):
    """
    Main processing function.
    
    Args:
        input_path: Path to input raster file (model output)
        output_path: Path where processed output should be saved
        parameters: Dictionary of processor parameters from the API
    """
    try:
        logger.info(f"Processing {input_path}")
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
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(processed_data)
        
        logger.info(f"Successfully processed and saved to {output_path}")
        
        # Return success status
        result = {
            "status": "success",
            "message": f"Processing completed with threshold {threshold}",
            "output_path": output_path,
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
    python script.py <input_path> <output_path> <parameters_json>
    """
    
    if len(sys.argv) < 4:
        logger.error("Usage: python script.py <input_path> <output_path> <parameters_json>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    parameters_json = sys.argv[3]
    
    # Parse parameters
    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse parameters JSON: {e}")
        parameters = {}
    
    # Run main processing
    exit_code = main(input_path, output_path, parameters)
    sys.exit(exit_code)
```

**Key Points:**
- The processor is a standalone Python script executed as `python script.py`
- The script receives arbitrary arguments via command line: e.g `input_path`, `output_path`, `threshold` ... 
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
GET /api/v2/generic-processor/{generic_processor_id}
```

#### Example

```bash
curl -X GET "https://your-api-url/api/v2/generic-processor/b1c40d04-1d36-43ed-b733-5dc18aa45689" \
  -H "Authorization: Bearer YOUR_API_KEY"
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
GET /api/v2/generic-processor
```

#### Example

```bash
curl -X GET "https://your-api-url/api/v2/generic-processor" \
  -H "Authorization: Bearer YOUR_API_KEY"
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
DELETE /api/v2/generic-processor/{generic_processor_id}
```

#### Example

```bash
curl -X DELETE "https://your-api-url/api/v2/generic-processor/b1c40d04-1d36-43ed-b733-5dc18aa45689" \
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
POST /api/v2/inference
```

#### Request Schema

```json
{
  "model_display_name": "string (optional if model_id provided)",
  "model_id": "uuid (optional if model_display_name provided)",
  "description": "string (optional)",
  "location": "string (optional)",
  "spatial_domain": {
    "bbox": [[lon_min, lat_min, lon_max, lat_max]],
    "polygons": [],
    "tiles": [],
    "urls": ["pre-signed-url"] # optional if bbox provided
  },
  "temporal_domain": ["YYYY-MM-DD"],
  "maxcc": 100,
  "fine_tuning_id": "string (optional)",
  "generic_processor_id": "uuid (optional)",
  "post_processing": {
    "cloud_masking": true,
    "snow_ice_masking": false
  }
}
```

#### Required Fields

- **Either** `model_id` **or** `model_display_name`
- `spatial_domain`: At least one of `bbox`, `polygons`, `tiles`, or `urls`
- `temporal_domain`: Array of date strings

#### Example: Basic Inference

```bash
curl -X POST "https://your-api-url/api/v2/inference" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_display_name": "flood-detection-model",
    "description": "Flood detection for Vienna region",
    "location": "Vienna, Austria",
    "spatial_domain": {
      "bbox": [[16.2, 48.1, 16.5, 48.3]]
    },
    "temporal_domain": ["2024-06-15"],
    "maxcc": 80
  }'
```

#### Example: Python

```python
import requests

url = "https://your-api-url/api/v2/inference"
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}

payload = {
    "model_display_name": "flood-detection-model",
    "description": "Flood detection for Vienna region",
    "location": "Vienna, Austria",
    "spatial_domain": {
        "bbox": [[16.2, 48.1, 16.5, 48.3]]
    },
    "temporal_domain": ["2024-06-15", "2024-06-16"],
    "maxcc": 80,
    "post_processing": {
        "cloud_masking": True,
        "permanent_water_masking": True
    }
}

response = requests.post(url, headers=headers, json=payload)
inference = response.json()
print(f"Inference ID: {inference['id']}")
print(f"Status: {inference['status']}")
```

#### Response

```json
{
  "id": "904d1e13-ddd2-415f-a963-120d16a240f0",
  "model_display_name": "flood-detection-model",
  "model_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "description": "Flood detection for Vienna region",
  "location": "Vienna, Austria",
  "spatial_domain": {
    "bbox": [[16.2, 48.1, 16.5, 48.3]]
  },
  "temporal_domain": ["2024-06-15"],
  "status": "PENDING",
  "created_at": "2026-01-21T10:30:00Z",
  "updated_at": "2026-01-21T10:30:00Z",
  "created_by": "user@example.com"
}
```

---

### Using Generic Processors in Inference

Integrate a custom Python processor into your inference pipeline by including the `generic_processor_id`.

#### Example

```bash
curl -X POST "https://your-api-url/api/v2/inference" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model_display_name": "flood-detection-model",
    "description": "Flood detection with custom cloud masking",
    "location": "Vienna, Austria",
    "spatial_domain": {
      "urls": ["https://example.com/satellite-image.tif"]
    },
    "temporal_domain": ["2024-06-15"],
    "generic_processor_id": "b1c40d04-1d36-43ed-b733-5dc18aa45689"
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
      "process_id": "terrakit-data-fetch",
      "step_number": 0
    },
    {
      "status": "WAITING",
      "process_id": "model-inference",
      "step_number": 1
    },
    {
      "status": "WAITING",
      "process_id": "generic-python-processor",
      "step_number": 2
    },
    {
      "status": "WAITING",
      "process_id": "push-to-geoserver",
      "step_number": 3
    }
  ]
}
```

---

### Monitoring Inference Status

#### Get Inference Details

```http
GET /api/v2/inference/{inference_id}
```

**Example:**

```bash
curl -X GET "https://your-api-url/api/v2/inference/904d1e13-ddd2-415f-a963-120d16a240f0" \
  -H "Authorization: Bearer YOUR_API_KEY"
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
- `RUNNING`: Currently executing
- `COMPLETED`: All tasks completed successfully
- `FAILED`: One or more tasks failed
- `STOPPED`: Manually stopped

#### Get Inference Tasks

```http
GET /api/v2/inference/{inference_id}/tasks
```

**Example:**

```bash
curl -X GET "https://your-api-url/api/v2/inference/904d1e13-ddd2-415f-a963-120d16a240f0/tasks" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Get Task Output URL

```http
GET /api/v2/tasks/{task_id}/output-url
```

**Example:**

```bash
curl -X GET "https://your-api-url/api/v2/tasks/904d1e13-ddd2-415f-a963-120d16a240f0-tile-001/output-url" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

---

## Complete Workflow Example

This example demonstrates the complete workflow from creating a generic processor to running inference.

```python
import requests
import json
import time

# Configuration
API_URL = "https://your-api-url/api/v2"
API_KEY = "your_api_key_here"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Step 1: Create a generic processor
print("Step 1: Creating generic processor...")
processor_metadata = {
    "name": "custom_cloud_filter",
    "description": "Advanced cloud filtering",
    "processor_parameters": {
        "threshold": 85,
        "method": "scl_based"
    }
}

files = {
    'generic_processor_file': ('cloud_filter.py', open('cloud_filter.py', 'rb'))
}
data = {
    'generic_processor': json.dumps(processor_metadata)
}

response = requests.post(
    f"{API_URL}/generic-processor",
    headers={"Authorization": f"Bearer {API_KEY}"},
    files=files,
    data=data
)
processor = response.json()
processor_id = processor['id']
print(f"✓ Processor created: {processor_id}")

# Step 2: Create inference with the generic processor
print("\nStep 2: Creating inference job...")
inference_payload = {
    "model_display_name": "flood-segmentation-v2",
    "description": "Flood detection with custom cloud filtering",
    "location": "Danube River, Austria",
    "spatial_domain": {
        "bbox": [[16.2, 48.1, 16.5, 48.3]]
    },
    "temporal_domain": ["2024-06-15"],
    "maxcc": 90,
    "generic_processor_id": processor_id
}

response = requests.post(
    f"{API_URL}/inference",
    headers=headers,
    json=inference_payload
)
inference = response.json()
inference_id = inference['id']
print(f"✓ Inference created: {inference_id}")

# Step 3: Monitor inference progress
print("\nStep 3: Monitoring inference progress...")
while True:
    response = requests.get(
        f"{API_URL}/inference/{inference_id}",
        headers=headers
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
        f"{API_URL}/inference/{inference_id}/tasks",
        headers=headers
    )
    tasks = response.json()
    
    for task in tasks['tasks']:
        if task['status'] == 'FINISHED':
            task_id = task['task_id']
            response = requests.get(
                f"{API_URL}/tasks/{task_id}/output-url",
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


### Generic Processor Not Executing

**Problem**: Processor step is skipped or fails

**Solutions**:
- Verify processor ID is correct
- Check processor status is FINISHED
- Ensure processor file is valid Python file
- Review processor logs for errors
- Ensure that the python file has an entrypoint `__main__`

---

## API Reference Summary

### Generic Processor Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/generic-processor` | Create processor |
| GET | `/api/v2/generic-processor/{id}` | Get processor |
| GET | `/api/v2/generic-processor` | List processors |
| DELETE | `/api/v2/generic-processor/{id}` | Delete processor |

### Inference Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v2/inference` | Create inference |
| GET | `/api/v2/inference/{id}` | Get inference |
| GET | `/api/v2/inference` | List inferences |
| GET | `/api/v2/inference/{id}/tasks` | Get tasks |
| DELETE | `/api/v2/inference/{id}` | Delete inference |

### Task Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v2/tasks/{task_id}/output-url` | Get output URL |
| GET | `/api/v2/tasks/{task_id}/logs/{step_id}` | Get logs |

---

## Related Documentation

- [GEOStudio Core README](./README.md)
- [API Documentation](https://your-api-url/docs)
- [Contributing Guide](./CONTRIBUTING.md)

---

## Support

For help:
- Open an issue on [GitHub](https://github.com/terrastackai/geospatial-studio-core/issues)
- Review [Contributing Guide](./CONTRIBUTING.md)
- Check [Code of Conduct](./CODE_OF_CONDUCT.md)