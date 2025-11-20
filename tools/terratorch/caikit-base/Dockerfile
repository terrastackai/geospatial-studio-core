# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

###############################################################################
# This Docker file defines layers with:
# - Runtime + test dependencies installed for local development
# - The extension wheel built and tested
# - The extension wheel installed for distribution
#
# It's meant to be run from a variety of base OS + Python install combinations, so no OS-specific
# instructions should go in here. (apt-get, dnf, yum...)
################################################################################

## Stage 1: Base Image with Conda Environment #################################
ARG PYTHON_TAG=py311
ARG OS=ubi8
ARG BASE_IMAGE_TAG=latest
ARG SOURCE_DIR="geospatial_extension"

## Conda Environment Layer
FROM us.icr.io/gfmaas/geostudio-tt-conda:latest as base

# Install necessary packages for groupadd and useradd.
RUN microdnf install -y shadow-utils git && microdnf clean all && \
    groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -ms /bin/bash appuser && \
    chown -R appuser:appgroup /app

# Set default working directory and change ownership
WORKDIR /app

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# speed up pip installs by caching a directory across builds
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
RUN --mount=type=cache,target=/root/.cache \
    pip3 install -U pip && \
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu118 && \
    # pip install terratorch==0.99.8 && \
    pip install terratorch==1.1 && \
    # pip install git+https://git@github.com/IBM/terratorch.git@hot_fix-1fa07cd8 && \
    # Pin terratorch dependencies
    pip uninstall -y torchmetrics && \
    pip install torchmetrics==1.3.1 lightning==2.4.0 && \
    # Fixing albucore to 0.0.16 because of the error below in albucore==0.0.17
    # ImportError: cannot import name 'preserve_channel_dim' from 'albucore.utils'
    pip install albucore==0.0.16

## Stage 2: Development Layer ###########################################################
FROM base as runtime_dependencies

# Copy in the GeoSpatial environment
COPY --from=base /opt/miniforge/ /opt/miniforge/
ENV PATH=/opt/miniforge/bin:$PATH

COPY webhooks /opt/miniforge/lib/python3.11/site-packages/webhooks

RUN ln -s /usr/bin/pip3.8 /usr/bin/pip
RUN pip install -U pip setuptools wheel

### Stage 3: Application Base #######################################
FROM runtime_dependencies as caikit-base

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

### Stage 4: Go Toolset and Protobuf ################################
FROM registry.access.redhat.com/ubi8/go-toolset:latest as gateway-base

USER root
# Upgrade to latest Go to resolve vulnerabilities
ARG GO_VERSION=1.23.7
ENV GOROOT=/usr/local/go

RUN dnf install -y unzip tar gzip python3.8 python3-pip && \
    dnf clean all

RUN curl -fsSL https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz | tar -C /usr/local -xzf - \
    && rm -rf /usr/local/go/bin/go1.*

ARG PROTOBUF_VERSION=30.0
RUN curl -LO https://github.com/protocolbuffers/protobuf/releases/download/v${PROTOBUF_VERSION}/protoc-${PROTOBUF_VERSION}-linux-x86_64.zip && \
    unzip protoc-30.0-linux-x86_64.zip -d /protoc && \
    rm protoc-30.0-linux-x86_64.zip

ENV GOPATH=/opt/app-root/src/go
ENV PATH=$GOROOT/bin:$GOPATH/bin:/protoc/bin:${PATH}

### Stage 5: Build Protos ###########################################
FROM caikit-base as build-protos

ARG SOURCE_DIR

COPY ${SOURCE_DIR} ${SOURCE_DIR}

# Caikit generates the proto files dymanically to be used by the grpc-gateway-wrapper
RUN RUNTIME_LIBRARY=${SOURCE_DIR} python3 -m caikit.runtime.dump_services protos/

##############################################
# Stage 6: Gateway build
FROM gateway-base as gateway-build

COPY --from=build-protos /app/protos /app/protos

ARG GRPC_METADATA="mm-model-id"

RUN pip3.8 install grpc-gateway-wrapper && \
    grpc-gateway-wrapper \
    # --proto_files $(cat proto_file_list.txt) \
    --proto_files /app/protos/geospatialextensionservice.proto /app/protos/caikit.runtime.GeospatialExtension.geospatialrequest.proto /app/protos/caikit_data_model.imageresult.proto /app/protos/caikit_data_model.runtime.modelpointer.proto /app/protos/caikit_data_model.common.producerid.proto \
    --metadata ${GRPC_METADATA} \
    --output_dir . \
    --install_deps

### Stage 7: Runtime Server Release ###############################
FROM caikit-base as runtime_server_release

ARG SOURCE_DIR

COPY ${SOURCE_DIR} ${SOURCE_DIR}
COPY ${SOURCE_DIR}/config/config.yml .

# Copy the gateway
COPY --from=gateway-build /opt/app-root/src/app /gateway
COPY --from=gateway-build /opt/app-root/src/swagger /swagger
COPY --from=gateway-build /app/protos /protos

RUN mkdir -p /app/models

COPY swagger /tmp/swagger

ENV CONFIG_FILES=${SOURCE_DIR}/config/config.yml
EXPOSE 8080 8085

RUN chmod -R +rx /app
COPY runtime_template/run_with_gateway.sh /run_with_gateway.sh
COPY runtime_template/start_runtime.py /app/start_runtime.py
COPY runtime_template/readiness_check.py /app/readiness_check.py

RUN chmod +rx /run_with_gateway.sh && \
    chown appuser:appgroup /run_with_gateway.sh /app/readiness_check.py

COPY /cleanup/cleanup.py /cleanup/cleanup.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +rx /entrypoint.sh && \
    chown appuser:appgroup /entrypoint.sh /cleanup/cleanup.py

# Install Tini (lightweight init system)
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini && \
    chown -R appuser:appgroup /app /tini

USER appuser

# Run both the gRPC server and REST gateway by default.
CMD ["/tini", "/run_with_gateway.sh"]
