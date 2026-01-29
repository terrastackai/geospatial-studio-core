# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime as base

USER root

RUN apt-get update && apt-get install -y curl git

RUN mamba install -y -c conda-forge \
    libspatialindex gdal libgdal libgdal-arrow-parquet "urllib3<2" h5netcdf \
    # Ensure cryptography >= 43.0.1 is installed, resolves: https://github.com/pyca/cryptography/security/advisories/GHSA-h4gh-qq45-vh27
    && mamba install -y 'cryptography>=43.0.1'\
    && mamba clean -afy

# Download and install kubectl
RUN true && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    chmod +x ./kubectl && \
    mv kubectl /usr/local/bin/ && \
    mkdir ~/.kube && \
    chmod 777 ~/.kube

WORKDIR /terratorch

# Install Python dependencies
RUN --mount=type=cache,target=/root/.cache \
    pip install --timeout=7200 terratorch==1.1 && \
    # Fix for terramind models
    pip install diffusers<=0.34.0

# Create a non-root user with a UID and GID greater than 10000
RUN groupadd -g 10001 appgroup && \
    useradd -r -u 10001 -g appgroup -ms /bin/bash appuser && \
    chown -R appuser:appgroup /terratorch

RUN mkdir /.cache
RUN mkdir /.local
RUN mkdir /.jupyter
RUN chmod 777 /.cache
RUN chmod 777 /.local
RUN chmod 777 /.jupyter
# Temporary fix requiring older torchgeo (can be removed in future)
RUN pip install torchgeo==0.6.2

# Setup UTF-8 locale
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LC_CTYPE=C.UTF-8

USER appuser

# Image default command
CMD ["/bin/bash"]
