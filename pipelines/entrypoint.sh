#!/bin/bash
set -e

if [ "$APP_MODE" = "v" ]; then
    echo "Starting in Standard TerraTorch mode..."

    MODEL_NAME=${MODEL_NAME:-"ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"}
    PLUGIN_NAME=${PLUGIN_NAME:-"terratorch_segmentation"}
    HF_HOME=${HF_HOME:-"/workspace/models/huggingface_cache"}

    echo "=========================================="
    echo "vLLM Prithvi Flood Detection Server"
    echo "=========================================="
    echo "Model: $MODEL_NAME"
    echo "Plugin: $PLUGIN_NAME"
    echo "Cache: $HF_HOME"
    echo "=========================================="
    echo ""
    # 1. Clear Hugging Face lock files
    # This prevents the 'OSError: PermissionError' if a previous pod crashed
    echo "Checking for stale lock files in /workspace/models..."
    find /workspace/models -name "*.lock" -delete

    # 2. Verify GPU availability
    # This logs the GPU status to help debug 'Pending' or 'Runtime' errors
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv
    else
        echo "WARNING: nvidia-smi not found. GPU may not be accessible."
    fi
    echo ""
    echo "Downloading model (first run may take several minutes)..."
    vllm serve "$MODEL_NAME" \
        --model-impl terratorch \
        --runner pooling \
        --convert embed \
        --trust-remote-code \
        --skip-tokenizer-init \
        --enforce-eager \
        --io-processor-plugin "$PLUGIN_NAME" \
        --enable-mm-embeds \
        --host 0.0.0.0 \
        --port 8000 \
        --dtype float16 \
        --gpu-memory-utilization 0.7 \
        --max-model-len 2048 \
        --max-num-seqs 4

else
    echo "Starting in Standard TerraTorch mode..."
    exec python /opt/app-root/src/orchestrate_wrapper.py

fi
