# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0

###############################################################################
# This Dockerfile sets up the geospatial studio core service to interact
# with the data and inference services
###############################################################################
ARG IMAGE_NAME=registry.access.redhat.com/ubi9/python-311

FROM ${IMAGE_NAME}:latest AS virtualenv

# hadolint ignore=DL3002
USER root

# Update base packages for security with specific CVE fixes
# Note: rasterio will use pre-built wheels that bundle GDAL, avoiding the need for system GDAL
RUN dnf update -y && \
    dnf upgrade -y \
        openssl-libs \
        openssl \
        openssl-devel \
        expat \
        expat-devel \
        mod_http2 \
        rsync \
        openssl-fips-provider \
    && dnf clean all && \
    rm -rf /var/cache/dnf

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV UV_PROJECT_ENVIRONMENT=/opt/app-root/

WORKDIR /app

COPY uv.lock pyproject.toml LICENSE ./
RUN uv sync --frozen --no-dev --no-editable && \
    # Temporary fix for a runtime bug introduced by polars
    pip uninstall -y polars polars-runtime-32

FROM ${IMAGE_NAME}:latest AS download_stage

# hadolint ignore=DL3002
USER root

WORKDIR /downloads

# Install kubectl v1.33.0 (latest stable with security fixes for Go dependencies)
RUN curl -fsSL --retry 5 --retry-delay 3 --retry-all-errors \
    -o ./kubectl \
    "https://dl.k8s.io/release/v1.33.0/bin/linux/amd64/kubectl" && \
    chmod +x ./kubectl

RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.44.6/yq_linux_amd64 -o /usr/local/bin/yq && \
    chmod +x /usr/local/bin/yq

# Install Helm v3.17.1 (latest version with Go security fixes)
ARG HELM_VERSION=v3.17.1
ADD https://get.helm.sh/helm-${HELM_VERSION}-linux-amd64.tar.gz /tmp
RUN tar -zxvf /tmp/helm-${HELM_VERSION}-linux-amd64.tar.gz -C /tmp && \
    mv /tmp/linux-amd64/helm /usr/local/bin/helm && \
    chmod +x /usr/local/bin/helm && \
    rm -rf /tmp/*

FROM ${IMAGE_NAME}:latest

WORKDIR /app

COPY --from=download_stage /downloads/kubectl /usr/local/bin/
COPY --from=download_stage /usr/local/bin/yq /usr/local/bin/
COPY --from=download_stage /usr/local/bin/helm /usr/local/bin/
COPY --from=virtualenv /opt/app-root/lib/python3.11/site-packages /opt/app-root/lib/python3.11/site-packages
COPY --from=virtualenv /opt/app-root/bin /opt/app-root/bin

RUN true && \
    mkdir ~/.kube && \
    chmod 777 ~/.kube

COPY alembic.ini logging.ini /app/
COPY ./gfmstudio/ /app/gfmstudio

USER root
RUN chmod +x /app/gfmstudio/amo/scripts/*

ENV UID=10001 \
    USER=appuser

# hadolint ignore=SC2086
USER root
RUN groupadd -g ${UID} -r ${USER} \
    && useradd -l -u ${UID} -r -g ${USER} ${USER} -ms /bin/bash \
    && mkdir -p /home/${USER}/.local \
    && mkdir -p /home/${USER}/.cache \
    && mkdir -p /app/amo \
    && mkdir -p /geotunes/tune-tasks \
    && mkdir -p /opt/app-root/src/.config/sentinelhub \
    && chown -R ${USER}:${USER} /home/${USER} /app/amo /geotunes/ /opt/app-root/src/.config/sentinelhub \
    && chmod -R 777 /app/amo \
    && chmod -R 777 /opt/app-root/src/.config/sentinelhub

USER ${USER}

CMD ["uvicorn", "gfmstudio.main:app", "--host", "0.0.0.0", "--port", "8080",  "--loop", "asyncio", "--log-config", "logging.ini"]
