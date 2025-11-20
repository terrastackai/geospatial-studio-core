#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0




# Ensure MODEL_ID is set
if [ -z "$MODEL_ID" ]; then
    echo "MODEL_ID environment variable is not set."
    exit 1
fi

# Set the new chart name
new_chart_name="${MODEL_ID}-chart"

# Path to the original and new chart directories
original_chart_dir="/app/gfmstudio/amo/a-model-00-name-chart"
new_chart_dir="/app/amo/$new_chart_name"

# Check if the original Helm chart directory exists
if [ -d "$original_chart_dir" ]; then
    echo "Found $original_chart_dir, proceeding with copying and modification..."

    # Copy the Helm chart directory
    echo "Copying helm chart"
    mkdir -p "$new_chart_dir"
    cp -R "$original_chart_dir/"* "$new_chart_dir/"
    cp -R "$original_chart_dir/.helmignore" "$new_chart_dir/"

    # Path to Chart.yaml and values.yaml files
    chart_file="$new_chart_dir/Chart.yaml"
    values_file="$new_chart_dir/values.yaml"

    # Rename the chart in Chart.yaml using yq
    echo "Renaming chart"
    yq e ".name = \"$new_chart_name\"" -i "$chart_file"

    # Update values.yaml using yq
    echo "Updating chart values"
    yq e ".modelFramework = \"$MODEL_FRAMEWORK\"" -i "$values_file"
    yq e ".modelID = \"$MODEL_ID\"" -i "$values_file"
    yq e ".namespace = \"$NAMESPACE\"" -i "$values_file"
    yq e ".resourceName = \"$RESOURCE_NAME\"" -i "$values_file"
    yq e ".deployImage = \"$DEPLOY_IMAGE\"" -i "$values_file"
    yq e ".serviceAccountName = \"$SERVICE_ACCOUNT_NAME\"" -i "$values_file"
    yq e ".imagePullSecretName = \"$IMAGE_PULL_SECRET_NAME\"" -i "$values_file"
    yq e ".cosBucketLocation = \"$COS_BUCKET_LOCATION\"" -i "$values_file"
    yq e ".cosBucketName = \"$COS_BUCKET_NAME\"" -i "$values_file"
    yq e ".cosEndpointURL = \"$COS_ENDPOINT_URL\"" -i "$values_file"
    yq e ".cosCredentialsSecretName = \"$COS_CREDENTIALS_SECRET_NAME\"" -i "$values_file"
    yq e ".gatewayURL = \"$GATEWAY_URL\"" -i "$values_file"
    yq e ".gfmaasApiKeySecretName = \"$GFMAAS_API_KEY_SECRET_NAME\"" -i "$values_file"
    yq e ".gfmaasApiCredKey = \"$GFMAAS_API_CRED_KEY\"" -i "$values_file"
    yq e ".configureInferenceToleration = \"$CONFIGURE_INFERENCE_TOLERATION\"" -i "$values_file"
    yq e ".inferenceSharedPvc = \"$AMO_INFERENCE_SHARED_PVC\"" -i "$values_file"
    # Update enabled flags for additional volumes and mounts
    #yq e ".additionalVolumeMounts.enabled = env(ADD_VOLUMES_MOUNTS)" -i "$values_file"
    #yq e ".additionalVolumes.enabled = env(ADD_VOLUMES_MOUNTS)" -i "$values_file"
    #yq e ".rcloneCOSToMount = \"$RCLONE_COS_TO_MOUNT\"" -i "$values_file"

    echo "Updated values in $values_file and chart name in $chart_file"

    # Lint the new chart to check for errors
    helm lint "$new_chart_dir"

else
    echo "Directory $original_chart_dir does not exist."
    exit 1
fi
