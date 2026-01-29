#!/bin/bash

if [ "$APP_MODE" = "vllm" ]; then
    # vLLM healthcheck: check the web endpoint
    curl -f http://localhost:8000/health || exit 1
else
    # TerraTorch healthcheck: check if the package is importable/functional
    python -c "import terratorch; print('OK')" || exit 1
fi
