# MLflow

This section contains dockerfiles related to mlflow

## 1. MLflow

Configuration for MLflow

### Building the docker image

Change directory to the MLflow Dockerfile

    cd tools/mlflow

Build dockerfile, you can change the image name and tag `abc.xy/namespace/mlflow:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/mlflow:tag .

### Running the docker image

    docker run --name mlflow --rm abc.xy/namespace/mlflow:tag

## 2. MLflow PG Notify

Configuration for MLflow postgres notification to the studio

### Building the docker image

Change directory to the MLflow notify Dockerfile

    cd tools/mlflow

Build dockerfile, you can change the image name and tag `abc.xy/namespace/mlflow-notify:tag`.

    docker buildx build -f Dockerfile --platform=linux/amd64 -t abc.xy/namespace/mlflow-notify:tag .

### Running the docker image
The MLflow image does not expose a port as it runs as a sidecar of the main core api container. Also you need to expose the `MLFLOW_DATABASE_URI` and `DATABASE_URI` as environment variables.

    docker run --name mlflow-notify --rm -e MLFLOW_DATABASE_URI=xxx -e DATABASE_URI=xxx abc.xy/namespace/mlflow-notify:tag
