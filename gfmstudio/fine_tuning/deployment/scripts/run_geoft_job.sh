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
export IMAGE_PULL_SECRET=${9:-ris-private-registry}
export RESOURCE_LIMIT_CPU=${10:-10}
export RESOURCE_LIMIT_Memory=${11:-32}
export RESOURCE_LIMIT_GPU=${12:-1}
export RESOURCE_REQUEST_CPU=${13:-6}
export RESOURCE_REQUEST_Memory=${14:-24}
export RESOURCE_REQUEST_GPU=${15:-1}
export RUN_TERRATORCH_TEST=${16}
export NODE_AFFINITY=${17}

# Replace the variable and properly indent the content
sed '/\${TUNING_CONFIG_YAML}/{
    s/.*//
    r /dev/stdin
}' "$tuning_jobs_deployment_template" > "/tmp/$output_manifest_yaml" <<EOF
$(echo "$config_content" | sed 's/^/    /')
EOF

envsubst < "/tmp/$output_manifest_yaml" > "/tmp/${output_manifest_yaml}.tmp" \
    && mv "/tmp/${output_manifest_yaml}.tmp" "/tmp/$output_manifest_yaml"

# kubectl apply --dry-run=client -f "/tmp/$output_manifest_yaml"
kubectl apply -f "/tmp/$output_manifest_yaml"
