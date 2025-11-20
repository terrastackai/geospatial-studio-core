# UBI Base image for Terratorch

This section the base dockerfile for finetuning with terratorch architecture

## Building the docker image

Change directory to the ubi-base Dockerfile

    cd tools/terratorch/ubi-base

Build dockerfile, you can change the image name and tag `abc.xy/namespace/ubi-base:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/ubi-base:tag .

## Running the docker image

    docker run --name ubi-base --rm abc.xy/namespace/ubi-base:tag

### Submitting task directly to a pod in OC

```sh
terratorch fit -c /working/sen1floods11_swin_studio_update.yaml
```
