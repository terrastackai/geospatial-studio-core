# Terratorch Base image for Pipelines

This section contains the base dockerfile for terratorch inference pipeline step

## Building the docker image

Change directory to the terratorch-base Dockerfile

    cd tools/terratorch-base

Build dockerfile, you can change the image name and tag `abc.xy/namespace/terratorch-base:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/terratorch-base:tag .

## Running the docker image

    docker run --name terratorch-base --rm abc.xy/namespace/terratorch-base:tag
