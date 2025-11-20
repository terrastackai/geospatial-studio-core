#!/bin/bash

# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0




# Script expects pre-signed URL and OUTPUT_FILE
if [ -z "$URL" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Missing URL or OUTPUT_FILE environment variables"
  exit 1
fi

# Perform the download
curl -o "${OUTPUT_FILE}" "${URL}"
echo "Download completed for ${OUTPUT_FILE}"

# Exit the script
exit 0
