#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0




# Check input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <helm-chart-directory>"
    exit 1
fi

chart_dir=$1

# Set a release name for template rendering (can be arbitrary for local rendering)
release_name="test-release"

# Check if the Helm chart directory exists
if [ ! -d "$chart_dir" ]; then
    echo "Directory $chart_dir does not exist."
    exit 1
fi

echo "Rendering YAML for $chart_dir"
# Ask user if they want to proceed with rendering
read -p "Do you want to render the YAML? (y/n): " render_confirm
if [[ $render_confirm =~ ^[Yy]$ ]]; then
    # Ask user where they want to render the YAML
    echo "Select the output method:"
    echo "1. Console"
    echo "2. Output to a file"
    read -p "Enter your choice (1 or 2): " output_choice
    
    if [ "$output_choice" == "1" ]; then
        # Render to console
        helm template "$release_name" "$chart_dir" --show-only templates/main-chart-template.yaml | more
    elif [ "$output_choice" == "2" ]; then
        # Ask for the output file name
        read -p "Enter the output file name: " file_name
        # Render to the specified file
        helm template "$release_name" "$chart_dir" --show-only templates/main-chart-template.yaml > "$file_name"
        echo "Output saved to $file_name"
    else
        echo "Invalid choice. Exiting."
        exit 1
    fi
else
    echo "Rendering canceled."
fi
