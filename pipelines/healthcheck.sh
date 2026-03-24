#!/bin/bash

set -e

PORT=${PORT:-8000}
HOST=${HOST:-"localhost"}
HEALTH_URL="http://${HOST}:${PORT}/health"

# Try to get health status
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo "vLLM server is healthy (HTTP $HTTP_CODE)"
    exit 0
else
    echo "vLLM server health check failed (HTTP $HTTP_CODE)"
    exit 1
fi
