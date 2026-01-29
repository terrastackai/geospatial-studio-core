# GDAL Base image for Pipelines

This section contains the gdal base dockerfile for pre and post-processing pipeline step

## Building the docker image

Change directory to the gdal-base Dockerfile

    cd tools/gdal-base

Build dockerfile, you can change the image name and tag `abc.xy/namespace/gdal-base:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/gdal-base:tag .

## Running the docker image

    docker run --name gdal-base --rm abc.xy/namespace/gdal-base:tag
