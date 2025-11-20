#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0




# Enable debug mode to print all commands
set -x

# Check input
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <helm-chart-directory> <release-name> <namespace>"
    exit 1
fi

chart_dir=$1
release_name=$2
NAMESPACE=$3

# Ensure release_name and NAMESPACE are set
if [ -z "$release_name" ] || [ -z "$NAMESPACE" ]; then
    echo "Error: Release name and Namespace must be provided."
    exit 1
fi

# Log the environment and input variables
echo "Using Helm chart directory: $chart_dir"
echo "Release Name: $release_name"
echo "Namespace: $NAMESPACE"

# Check if the Helm chart directory exists
if [ ! -d "$chart_dir" ]; then
    echo "Directory $chart_dir does not exist."
    exit 1
fi

# Log the existing Helm releases
echo "Existing Helm releases in $NAMESPACE:"
helm list --namespace "$NAMESPACE" --debug

# Check if the release already exists using helm list
if helm list --namespace "$NAMESPACE" --filter "^${release_name}$" | grep -q "${release_name}"; then
    echo "Release found, upgrading..."
    helm upgrade "$release_name" "$chart_dir" -n "$NAMESPACE" --debug
else
    echo "Release not found, installing..."
    helm install "$release_name" "$chart_dir" -n "$NAMESPACE" --debug
fi

# Disable debug mode
set +x
