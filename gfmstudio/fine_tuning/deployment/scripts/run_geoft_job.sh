#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Check if argument is supplied
if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi

# Processes the input to match the desired format
# Escape special characters in the config
fine_tune_name=$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr '_' '-' | tr -d '[:space:]' | tr -cd '[:alnum:]-')
output_manifest_yaml="$fine_tune_name-deployment.yaml"
echo "Using FTUNE_NAME: $fine_tune_name"

# Check if the config file exists
config_yaml=$2
if [ ! -f "$config_yaml" ]; then
    echo "Error: Config file $config_yaml not found!"
    exit 1
fi

# Check if the deployment manifect template file exists
tuning_jobs_deployment_template=$3
if [ ! -f "$config_yaml" ]; then
    echo "Error: Config file $tuning_jobs_deployment_template not found!"
    exit 1
fi

# Read the config content from the file
config_content=$(<"$config_yaml")

export FTUNE_NAME="$fine_tune_name"
export JOB_UID=$(kubectl get jobs -o custom-columns=:metadata.uid --no-headers | head -n 1 || echo "dummy-uid")
export FT_API_KEY=$5
export TUNE_ID=$4
export FT_WEBHOOKS_ID=$6
export FT_WEBHOOKS_URL=$7
export FTUNING_RUNTIME_IMAGE=$8
export IMAGE_PULL_SECRET=${9}
export RESOURCE_LIMIT_CPU=${10:-10}
export RESOURCE_LIMIT_Memory=${11:-32}
export RESOURCE_LIMIT_GPU=${12:-1}
export RESOURCE_REQUEST_CPU=${13:-6}
export RESOURCE_REQUEST_Memory=${14:-24}
export RESOURCE_REQUEST_GPU=${15:-1}
export RUN_TERRATORCH_TEST=${16}
export APPEND_SECURITY_CONTEXT=${17:-false}
export SECURITY_CONTEXT_FSGROUP=${18:-2000}
export NODE_AFFINITY=${19}
export HF_HOME=${20:-/terratorch/gfm_models}
export TRANSFORMERS_CACHE=${21:-/terratorch/gfm_models}
export HF_HUB_OFFLINE=${22:-}
export TRANSFORMERS_OFFLINE=${23:-}

# Replace the variable and properly indent the content
sed '/\${TUNING_CONFIG_YAML}/{
    s/.*//
    r /dev/stdin
}' "$tuning_jobs_deployment_template" > "/tmp/$output_manifest_yaml" <<EOF
$(echo "$config_content" | sed 's/^/    /')
EOF

envsubst < "/tmp/$output_manifest_yaml" > "/tmp/${output_manifest_yaml}.tmp" \
    && mv "/tmp/${output_manifest_yaml}.tmp" "/tmp/$output_manifest_yaml"

# Remove imagePullSecrets section if IMAGE_PULL_SECRET is empty
if [ -z "$IMAGE_PULL_SECRET" ]; then
    echo "IMAGE_PULL_SECRET is empty, removing imagePullSecrets from manifest"
    # Remove the imagePullSecrets section (handles both single-line and multi-line formats)
    sed -i '/imagePullSecrets:/,/^[^ ]/{ /imagePullSecrets:/d; /- name:/d; /^[^ ]/!d; }' "/tmp/$output_manifest_yaml"
fi

# Add security context if APPEND_SECURITY_CONTEXT is true
if [ "$APPEND_SECURITY_CONTEXT" = "true" ] || [ "$APPEND_SECURITY_CONTEXT" = "True" ]; then
    echo "Adding pod security context with fsGroup: $SECURITY_CONTEXT_FSGROUP"
    sed -i '/serviceAccountName: api-gateway-sa/a\      securityContext:\n        fsGroup: '"$SECURITY_CONTEXT_FSGROUP"'\n        fsGroupChangePolicy: "OnRootMismatch"' "/tmp/$output_manifest_yaml"
fi

# kubectl apply --dry-run=client -f "/tmp/$output_manifest_yaml"
kubectl apply -f "/tmp/$output_manifest_yaml"
