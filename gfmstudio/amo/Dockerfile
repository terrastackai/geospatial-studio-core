# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


# Use Red Hat Universal Base Image (UBI) minimal image
FROM registry.access.redhat.com/ubi8/ubi-minimal:8.10-1052.1724178568

# Set label metadata
LABEL org.opencontainers.image.title="Curl Download Container"
LABEL org.opencontainers.image.description="This container is designed to perform curl downloads for model artifacts in the form of pre-signed artifacts. Additionally, yq is used create a system config file. To be used on Openshift Cluster."
LABEL org.opencontainers.image.version="1.2.0"

# Update system and only install curl
RUN microdnf update -y && \
    microdnf install -y curl wget && \
    microdnf clean all

RUN wget https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 -O /usr/bin/yq && \
    chmod +x /usr/bin/yq


# Prepare directory to mount model COS bucket
RUN mkdir -p /data && \
    chmod -R 777 /data

# Set work directory
WORKDIR /data

# Add the download script
COPY scripts/model-artifact-load.sh /app/scripts/model-artifact-load.sh
RUN chmod +x /app/scripts/model-artifact-load.sh

# Entrypoint of script : Download and exit
ENTRYPOINT ["/app/scripts/model-artifact-load.sh"]