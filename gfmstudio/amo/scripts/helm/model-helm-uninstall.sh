#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0




# Check input
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <release-name> <model_id> <namespace>"
    exit 1
fi

release_name=$1
model_id=$2
namespace=$3

# Enable debug mode 
set -x

# Uninstall Helm release; if helm release is not available continue to remove other assets
helm uninstall "$release_name" -n "$namespace" || true

# Delete any jobs
kubectl delete job --selector=amo="$model_id" -n "$namespace"

# Delete any lingering deployment created by helm or manually
echo "Waiting for all deployment to terminate..."
kubectl delete deployment --selector=amo="$model_id" -n "$namespace"

# Delete any lingering services created by Helm or manually
kubectl delete svc --selector=amo="$model_id" -n "$namespace"

# Delete any routes if using OpenShift or similar
kubectl delete route --selector=amo="$model_id" -n "$namespace"

# Delete any persistent volume claims
kubectl delete pvc --selector=amo="$model_id" -n "$namespace"

# Optional: Clean up any configmaps or secrets
kubectl delete configmaps --selector=amo="$model_id" -n "$namespace"
kubectl delete secrets --selector=amo="$model_id" -n "$namespace"
# Disable debug mode
set +x

echo "Cleanup complete for release $release_name in namespace $namespace."
