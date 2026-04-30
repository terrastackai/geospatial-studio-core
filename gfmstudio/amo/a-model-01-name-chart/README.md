# Automated Model Deployment Guide

This guide explains how to use the vllm-inference-server template for automated model deployments.

## Overview

The deployment system uses a template-based approach where:
- All services are prefixed with `gfm-amo-{MODEL_NAME}`
- KServe can be toggled on/off during deployment
- Resource requirements are configurable per deployment
- Model names are dynamically injected into the deployment

## Files

1. **vllm-inference-server-template.yaml** - The main template with placeholders
2. **deploy_model.py** - Python script for automated deployments
3. **vllm-inference-server-all-in-one.yaml** - Reference with Helm variables
4. **vllm-inference-server-standalone.yaml** - Example with filled values

## Template Placeholders

The template uses the following placeholders that must be replaced:

| Placeholder | Description | Example |
|------------|-------------|---------|
| `${MODEL_NAME}` | Model name (prefixed with gfm-amo-) | `my-model` |
| `${NAMESPACE}` | Kubernetes namespace | `geospatial-studio` |
| `${ENABLE_KSERVE}` | Enable KServe (true/false) | `false` |
| `${IMAGE_REPOSITORY}` | Container image repository | `us.icr.io/gfmaas/vllm-small` |
| `${IMAGE_TAG}` | Container image tag | `v0.0.6` |
| `${IMAGE_PULL_SECRET}` | Image pull secret name | `my-registry-secret` |
| `${SERVICE_ACCOUNT}` | Service account name | `default` |
| `${MODELS_PVC}` | Models storage PVC | `vllm-models-pvc` |
| `${INFERENCE_SHARED_PVC}` | Shared inference PVC | `inference-shared-pvc` |
| `${GPU_COUNT}` | Number of GPUs | `1` |
| `${CPU_LIMIT}` | CPU limit | `2000m` |
| `${MEMORY_LIMIT}` | Memory limit | `8Gi` |
| `${CPU_REQUEST}` | CPU request | `1000m` |
| `${MEMORY_REQUEST}` | Memory request | `4Gi` |

## Deployment Modes

### Standard Deployment (KServe Disabled)

Creates a standard Kubernetes Deployment with:
- Fixed replica count (default: 1)
- Always-on pods
- Direct service access

**Use when:**
- You need consistent availability
- Scale-to-zero is not required
- You want simpler networking

### KServe InferenceService (KServe Enabled)

Creates a KServe InferenceService with:
- Scale-to-zero capability (minReplicas: 0)
- Automatic scaling based on traffic
- Advanced serving features

**Use when:**
- You want to save resources with scale-to-zero
- You need automatic scaling
- KServe is installed in your cluster

## Usage Examples

### Using Python Script

#### 1. Deploy with Standard Deployment (Dry Run)

```bash
python deploy_model.py my-flood-model \
  --namespace geospatial-studio \
  --dry-run
```

#### 2. Deploy with KServe InferenceService

```bash
python deploy_model.py my-flood-model \
  --namespace geospatial-studio \
  --enable-kserve
```

#### 3. Deploy with Custom Resources

```bash
python deploy_model.py my-large-model \
  --namespace geospatial-studio \
  --gpu-count 2 \
  --memory-limit 16Gi \
  --cpu-limit 4000m \
  --memory-request 8Gi \
  --cpu-request 2000m
```

#### 4. Generate YAML File Without Deploying

```bash
python deploy_model.py my-model \
  --namespace geospatial-studio \
  --enable-kserve \
  --dry-run \
  --output my-model-deployment.yaml
```

### Using Shell Script (sed/envsubst)

```bash
#!/bin/bash

# Set variables
export MODEL_NAME="my-flood-model"
export NAMESPACE="geospatial-studio"
export ENABLE_KSERVE="false"
export IMAGE_REPOSITORY="us.icr.io/gfmaas/vllm-small"
export IMAGE_TAG="v0.0.6"
export IMAGE_PULL_SECRET="my-registry-secret"
export SERVICE_ACCOUNT="default"
export MODELS_PVC="vllm-models-pvc"
export INFERENCE_SHARED_PVC="inference-shared-pvc"
export GPU_COUNT="1"
export CPU_LIMIT="2000m"
export MEMORY_LIMIT="8Gi"
export CPU_REQUEST="1000m"
export MEMORY_REQUEST="4Gi"

# Generate YAML
envsubst < vllm-inference-server-template.yaml > deployment.yaml

# Filter based on KServe mode
if [ "$ENABLE_KSERVE" = "true" ]; then
  # Remove Deployment section, keep InferenceService
  sed -i '/# Deployment (used when ENABLE_KSERVE=false)/,/^---$/d' deployment.yaml
else
  # Remove InferenceService section, keep Deployment
  sed -i '/# InferenceService (used when ENABLE_KSERVE=true)/,/^# Made with Bob$/d' deployment.yaml
fi

# Apply
kubectl apply -f deployment.yaml
```

### Programmatic Integration (Python)

```python
from deploy_model import deploy_model

# Deploy a model programmatically
success = deploy_model(
    model_name="my-burn-scar-model",
    namespace="geospatial-studio",
    enable_kserve=True,
    image_repository="us.icr.io/gfmaas/vllm-small",
    image_tag="v0.0.6",
    gpu_count="1",
    memory_limit="8Gi",
    dry_run=False
)

if success:
    print("Model deployed successfully!")
else:
    print("Deployment failed!")
```

## Service Naming Convention

All deployed services follow this naming pattern:
- Service name: `gfm-amo-{MODEL_NAME}`
- Example: `gfm-amo-my-flood-model`

This ensures:
- Consistent naming across deployments
- Easy identification of automated deployments
- No naming conflicts with manual deployments

## Verification

After deployment, verify the resources:

```bash
# Check deployment/inferenceservice
kubectl get deployment -n geospatial-studio gfm-amo-{MODEL_NAME}
# OR
kubectl get inferenceservice -n geospatial-studio gfm-amo-{MODEL_NAME}

# Check service
kubectl get svc -n geospatial-studio gfm-amo-{MODEL_NAME}

# Check pods
kubectl get pods -n geospatial-studio -l app.kubernetes.io/name=gfm-amo-{MODEL_NAME}

# View logs
kubectl logs -n geospatial-studio -l app.kubernetes.io/name=gfm-amo-{MODEL_NAME}
```

## Cleanup

To remove a deployed model:

```bash
# Delete all resources for a model
kubectl delete all -n geospatial-studio -l app.kubernetes.io/name=gfm-amo-{MODEL_NAME}

# Or delete specific resources
kubectl delete deployment gfm-amo-{MODEL_NAME} -n geospatial-studio
kubectl delete svc gfm-amo-{MODEL_NAME} -n geospatial-studio
```

## Integration with Deployment Services

For automated deployment services, the recommended approach is:

1. **Store the template** in your deployment service
2. **Accept user inputs** for model name and KServe toggle
3. **Replace placeholders** using your preferred method (Python, Go, etc.)
4. **Filter resources** based on KServe mode
5. **Apply to cluster** using kubectl or Kubernetes client library

Example workflow:
```
User Request → Model Name + KServe Toggle
    ↓
Load Template
    ↓
Replace Placeholders
    ↓
Filter by KServe Mode
    ↓
Apply to Kubernetes
    ↓
Return Service URL: gfm-amo-{MODEL_NAME}.{NAMESPACE}.svc.cluster.local
```

## Troubleshooting

### Issue: Pods not starting

Check:
1. GPU availability: `kubectl describe node | grep nvidia.com/gpu`
2. Image pull secrets: `kubectl get secret {IMAGE_PULL_SECRET} -n {NAMESPACE}`
3. PVC status: `kubectl get pvc -n {NAMESPACE}`

### Issue: KServe InferenceService not working

Verify:
1. KServe is installed: `kubectl get crd inferenceservices.serving.kserve.io`
2. Knative Serving is running: `kubectl get pods -n knative-serving`
3. Check InferenceService status: `kubectl describe inferenceservice gfm-amo-{MODEL_NAME} -n {NAMESPACE}`

### Issue: Service not accessible

Check:
1. Service exists: `kubectl get svc gfm-amo-{MODEL_NAME} -n {NAMESPACE}`
2. Endpoints are ready: `kubectl get endpoints gfm-amo-{MODEL_NAME} -n {NAMESPACE}`
3. Pod is running: `kubectl get pods -n {NAMESPACE} -l app.kubernetes.io/name=gfm-amo-{MODEL_NAME}`
