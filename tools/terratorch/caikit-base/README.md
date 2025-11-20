# Caikit Base image for Terratorch

This section contains the code used in the deployment of the geospation foundation models inference service using caikit. The code is based on the [Caikit package](https://github.com/caikit/caikit) for serving models. This code exposes the server providing APIs in REST format.

### Project Structure

---

This section is structured as seen below and the main code for running this application is the `geospatial_extension`.

This folder follows the recommended caikit project structure. For more details see this [tutorial](https://caikit.github.io/website/docs/tutorial_appdev.html).

```
├── cleanup # scripts for cleanup in the pods.
├── geospatial_extension # main codebase
├── models # contains the model artifacts
│   ├── config.yml
│   ├── model_specific_config.yml/py
│   └── model_checkpoint.ckpt/pt
├── runtime_template # scripts to startup caikit
├── swagger # api docs
├── tests # unit & integration tests
└── webhooks # scripts to notify UI
```

## Usage

### Prerequisites

- git
- Access to the specific model frameworks repos i.e terratorch.
- docker
- Required python version by the model framework.

### Building the docker image

Change directory to the caikit-base Dockerfile

    cd tools/terratorch/caikit-base

Build dockerfile, you can change the image name and tag `abc.xy/namespace/caikit-base:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/caikit-base:tag .

### Running the docker image

    docker run --name caikit-base --rm -p 3000:3000 abc.xy/namespace/caikit-base:tag
