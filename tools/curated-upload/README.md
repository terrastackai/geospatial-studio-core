# Curated dataset onboarding

This section contains the dockerfile used to creat container images for running fine-tuning dataset onboarding

## Building the docker image

Change directory to the curated-upload Dockerfile

    cd tools/curated-upload

Build dockerfile, you can change the image name and tag `abc.xy/namespace/curated-upload:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/curated-upload:tag .

## Running the docker image

    docker run --name curated-upload --rm abc.xy/namespace/curated-upload:tag

