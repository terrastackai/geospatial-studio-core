# vLLM Base image for Terratorch

This section contains the code used in the deployment of the geospation foundation models inference service using vLLM.

## Building the docker image

Change directory to the vllm-base Dockerfile

    cd tools/terratorch/vllm-base

Build dockerfile, you can change the image name and tag `abc.xy/namespace/vllm-base:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/vllm-base:tag .

## Running the docker image

    docker run --name vllm-base --rm abc.xy/namespace/vllm-base:tag
